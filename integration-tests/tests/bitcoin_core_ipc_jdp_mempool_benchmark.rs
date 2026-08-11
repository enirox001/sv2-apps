//! Manual end-to-end latency benchmark for the Bitcoin Core IPC JDP mempool mirror.
//!
//! Each sample prepares a signed transaction, starts a monotonic clock immediately before the
//! `sendrawtransaction` RPC, and probes JDP with its wtxid until JDP no longer reports the
//! transaction as missing. The measured duration therefore includes Bitcoin Core mempool
//! admission, `waitNext`, `getBlock`, deserialization, the mirror update, and one JDP probe.
//!
//! Run from `integration-tests/` with, for example:
//! `SV2_JDP_BENCH_SAMPLES=20 RUST_LOG=warn cargo test --release --test
//! bitcoin_core_ipc_jdp_mempool_benchmark jdp_mempool_mirror_latency_v31x -- --ignored --nocapture
//! --test-threads=1`.

use async_channel::Sender;
use integration_tests_sv2::{
    start_bitcoin_core, start_tracing, template_provider::DifficultyLevel,
};
use std::time::{Duration, Instant};
use stratum_apps::{
    bitcoin_core_sv2::{
        CancellationToken,
        runtime_api::{
            BitcoinCoreVersion,
            job_declaration_protocol::{
                self,
                io::{JdRequest, JdResponse},
            },
        },
    },
    stratum_core::bitcoin::{
        Amount, Block, OutPoint, ScriptBuf, Sequence, Transaction, TxIn, TxOut, Witness, Wtxid,
        absolute::LockTime,
        block::{Version as BlockVersion, WitnessMerkleNode},
        hashes::Hash,
        transaction::Version as TxVersion,
    },
};

const DEFAULT_SAMPLES: usize = 10;
const PROBE_INTERVAL: Duration = Duration::from_millis(10);
const SAMPLE_TIMEOUT: Duration = Duration::from_secs(20);

#[tokio::test]
#[ignore = "manual benchmark: starts a real Bitcoin Core v31 node"]
async fn jdp_mempool_mirror_latency_v31x() {
    run_benchmark(BitcoinCoreVersion::V31X).await;
}

#[tokio::test]
#[ignore = "manual benchmark: starts a real Bitcoin Core v30 node"]
async fn jdp_mempool_mirror_latency_v30x() {
    run_benchmark(BitcoinCoreVersion::V30X).await;
}

async fn run_benchmark(version: BitcoinCoreVersion) {
    start_tracing();

    let sample_count = std::env::var("SV2_JDP_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(DEFAULT_SAMPLES);

    let bitcoin_core = start_bitcoin_core(DifficultyLevel::Low, version);
    // Mature enough regtest coinbase outputs for the wallet to fund benchmark transactions.
    bitcoin_core
        .fund_wallet()
        .expect("failed to fund the benchmark wallet");

    let next_height = bitcoin_core
        .get_blockchain_info()
        .expect("failed to get blockchain info")
        .blocks
        + 1;
    let next_height = u32::try_from(next_height).expect("next height should fit in u32");
    let (incoming_sender, incoming_receiver) = async_channel::unbounded::<JdRequest>();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let cancellation_token = CancellationToken::new();
    let cancellation_token_clone = cancellation_token.clone();
    let socket_path = bitcoin_core.ipc_socket_path();

    let jdp_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create Tokio runtime");
        let local_set = tokio::task::LocalSet::new();

        local_set.block_on(&runtime, async move {
            let jdp = job_declaration_protocol::new(
                version,
                socket_path,
                incoming_receiver,
                cancellation_token_clone,
                ready_tx,
            )
            .await
            .expect("failed to initialize BitcoinCoreSv2JDP");

            jdp.run().await;
        });
    });

    tokio::time::timeout(Duration::from_secs(30), ready_rx)
        .await
        .expect("timed out waiting for JDP readiness")
        .expect("JDP readiness channel dropped unexpectedly");

    println!(
        "\nJDP mempool mirror benchmark: version={version:?}, samples={sample_count}\n\
         measurement=sendrawtransaction start -> JDP no longer reports wtxid missing\n"
    );
    println!("sample\trpc_ms\tend_to_end_ms\tprobes\toutcome");

    let mut rpc_samples = Vec::with_capacity(sample_count);
    let mut end_to_end_samples = Vec::with_capacity(sample_count);
    let mut benchmark_wtxids = Vec::with_capacity(sample_count);

    for sample in 1..=sample_count {
        // Transaction construction and wallet signing are intentionally outside the timer.
        let transaction = bitcoin_core
            .create_signed_mempool_transaction()
            .expect("failed to prepare signed benchmark transaction");
        let wtxid = transaction.compute_wtxid();
        benchmark_wtxids.push(wtxid);

        let started = Instant::now();
        bitcoin_core
            .send_raw_transaction(&transaction)
            .expect("sendrawtransaction failed");
        let rpc_elapsed = started.elapsed();

        let (end_to_end_elapsed, probes, outcome) = wait_until_visible_to_jdp(
            &incoming_sender,
            next_height,
            &benchmark_wtxids,
            wtxid,
            started,
            sample,
        )
        .await;

        rpc_samples.push(rpc_elapsed);
        end_to_end_samples.push(end_to_end_elapsed);
        println!(
            "{sample}\t{:.3}\t{:.3}\t{probes}\t{outcome}",
            duration_ms(rpc_elapsed),
            duration_ms(end_to_end_elapsed),
        );
    }

    print_summary("sendrawtransaction RPC", &mut rpc_samples);
    print_summary("end-to-end mirror visibility", &mut end_to_end_samples);

    cancellation_token.cancel();
    jdp_thread
        .join()
        .expect("BitcoinCoreSv2JDP thread join should succeed");
}

async fn wait_until_visible_to_jdp(
    incoming_sender: &Sender<JdRequest>,
    next_height: u32,
    declared_wtxids: &[Wtxid],
    target_wtxid: Wtxid,
    started: Instant,
    sample: usize,
) -> (Duration, usize, &'static str) {
    let mut probes = 0;

    loop {
        assert!(
            started.elapsed() < SAMPLE_TIMEOUT,
            "sample {sample} timed out waiting for {target_wtxid} to reach the JDP mempool mirror"
        );

        probes += 1;
        let response = send_probe(incoming_sender, next_height, declared_wtxids).await;
        match response {
            JdResponse::MissingTransactions { missing_wtxids, .. } => {
                assert!(
                    missing_wtxids.contains(&target_wtxid),
                    "sample {sample} returned MissingTransactions without the benchmark wtxid"
                );
                tokio::time::sleep(PROBE_INTERVAL).await;
            }
            JdResponse::Success { .. } => {
                return (started.elapsed(), probes, "success");
            }
            JdResponse::Error { error_code, .. } => panic!(
                "sample {sample} reached the mirror but JDP rejected the probe: {error_code}"
            ),
        }
    }
}

async fn send_probe(
    incoming_sender: &Sender<JdRequest>,
    next_height: u32,
    declared_wtxids: &[Wtxid],
) -> JdResponse {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    incoming_sender
        .send(JdRequest::DeclareMiningJob {
            version: BlockVersion::from_consensus(0x2000_0000),
            coinbase_tx: build_valid_coinbase_tx(next_height, declared_wtxids),
            wtxid_list: declared_wtxids.to_vec(),
            missing_txs: vec![],
            response_tx,
        })
        .await
        .expect("failed to send JDP benchmark probe");

    tokio::time::timeout(Duration::from_secs(5), response_rx)
        .await
        .expect("timed out waiting for JDP benchmark probe")
        .expect("JDP benchmark probe response channel dropped")
}

fn print_summary(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let sum_ms: f64 = samples.iter().copied().map(duration_ms).sum();
    let mean_ms = sum_ms / samples.len() as f64;

    println!(
        "{label}: mean={mean_ms:.3} ms, p50={:.3} ms, p95={:.3} ms, p99={:.3} ms, min={:.3} ms, max={:.3} ms",
        duration_ms(percentile(samples, 50)),
        duration_ms(percentile(samples, 95)),
        duration_ms(percentile(samples, 99)),
        duration_ms(samples[0]),
        duration_ms(samples[samples.len() - 1]),
    );
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let index = (samples.len() * percentile).div_ceil(100).saturating_sub(1);
    samples[index]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn coinbase_script_sig_for_height(height: u32) -> ScriptBuf {
    let mut encoded_height = Vec::new();
    let mut value = height;

    while value > 0 {
        encoded_height.push((value & 0xff) as u8);
        value >>= 8;
    }

    if encoded_height.last().is_some_and(|byte| byte & 0x80 != 0) {
        encoded_height.push(0x00);
    }

    let mut script = Vec::with_capacity(1 + encoded_height.len());
    script.push(encoded_height.len() as u8);
    script.extend_from_slice(&encoded_height);
    ScriptBuf::from_bytes(script)
}

fn build_valid_coinbase_tx(next_height: u32, transaction_wtxids: &[Wtxid]) -> Transaction {
    let witness_hashes = std::iter::once(Wtxid::all_zeros().to_raw_hash())
        .chain(transaction_wtxids.iter().map(|wtxid| wtxid.to_raw_hash()));
    let witness_root: WitnessMerkleNode =
        stratum_apps::stratum_core::bitcoin::merkle_tree::calculate_root(witness_hashes)
            .expect("the non-empty witness tree always has a root")
            .into();
    let witness_reserved_value = [0_u8; 32];
    let witness_commitment =
        Block::compute_witness_commitment(&witness_root, &witness_reserved_value);
    let mut witness_commitment_script = vec![0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    witness_commitment_script.extend_from_slice(&witness_commitment.to_byte_array());

    Transaction {
        version: TxVersion::TWO,
        lock_time: LockTime::ZERO,
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            script_sig: coinbase_script_sig_for_height(next_height),
            sequence: Sequence::MAX,
            witness: Witness::from_slice(&[witness_reserved_value]),
        }],
        output: vec![
            TxOut {
                value: Amount::from_sat(0),
                script_pubkey: ScriptBuf::new(),
            },
            TxOut {
                value: Amount::from_sat(0),
                script_pubkey: ScriptBuf::from_bytes(witness_commitment_script),
            },
        ],
    }
}
