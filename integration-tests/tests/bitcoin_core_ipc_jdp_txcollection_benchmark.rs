//! Manual end-to-end latency benchmark for the Bitcoin Core v32 TxCollection JDP backend.
//!
//! Run from `integration-tests/`:
//! `BITCOIN_CORE_V32_BINARY=<bitcoin>/build/bin/bitcoin-node \
//!  SV2_JDP_BENCH_SAMPLES=20 RUST_LOG=warn cargo test --release --test \
//!  bitcoin_core_ipc_jdp_txcollection_benchmark -- --ignored --nocapture --test-threads=1`

use async_channel::Sender;
use integration_tests_sv2::{
    start_bitcoin_core, start_tracing, template_provider::DifficultyLevel,
};
use std::time::{Duration, Instant};
use stratum_apps::{
    bitcoin_core_sv2::common::{
        job_declaration_protocol::{
            self,
            io::{JdRequest, JdResponse},
            CancellationToken,
        },
        BitcoinCoreVersion,
    },
    stratum_core::bitcoin::{
        absolute::LockTime,
        block::{Version as BlockVersion, WitnessMerkleNode},
        consensus::deserialize,
        hashes::Hash,
        transaction::Version as TxVersion,
        Amount, Block, OutPoint, ScriptBuf, Sequence, Transaction, TxIn, TxOut, Witness, Wtxid,
    },
};

const DEFAULT_SAMPLES: usize = 20;

#[tokio::test]
#[ignore = "manual benchmark: requires a Bitcoin Core PR #35671 bitcoin-node"]
async fn jdp_txcollection_latency_v32x() {
    start_tracing();
    let sample_count = std::env::var("SV2_JDP_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(DEFAULT_SAMPLES);

    let bitcoin_core = start_bitcoin_core(DifficultyLevel::Low, BitcoinCoreVersion::V32X);
    let funded_address = bitcoin_core
        .fund_wallet_legacy()
        .expect("failed to fund benchmark wallet");
    let next_height = u32::try_from(
        bitcoin_core
            .get_blockchain_info()
            .expect("failed to get blockchain info")
            .blocks
            + 1,
    )
    .expect("next height should fit in u32");
    // Prepare these before the mempool-hit run spends the wallet's first mature coinbase.
    // Each call uses a fresh destination, so every sample has a distinct unknown wtxid.
    let missing_transactions: Vec<Transaction> = (0..sample_count)
        .map(|_| {
            let bytes = bitcoin_core
                .create_unbroadcast_legacy_transaction(&funded_address)
                .expect("failed to create unbroadcast transaction");
            deserialize(&bytes).expect("failed to decode transaction")
        })
        .collect();

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
                BitcoinCoreVersion::V32X,
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
        .expect("JDP readiness channel dropped");

    println!("\nTxCollection mempool-hit path ({sample_count} samples)");
    println!("sample\tsendraw_ms\tcollect_make_ms\tend_to_end_ms");
    let mut sendraw_samples = Vec::with_capacity(sample_count);
    let mut collect_samples = Vec::with_capacity(sample_count);
    let mut end_to_end_samples = Vec::with_capacity(sample_count);
    let mut mempool_wtxids = Vec::with_capacity(sample_count);
    for sample in 1..=sample_count {
        let transaction = bitcoin_core
            .create_signed_mempool_transaction()
            .expect("failed to prepare signed transaction");
        mempool_wtxids.push(transaction.compute_wtxid());
        let coinbase = build_valid_coinbase_tx(next_height, &mempool_wtxids);

        let started = Instant::now();
        bitcoin_core
            .send_raw_transaction(&transaction)
            .expect("sendrawtransaction failed");
        let sendraw_elapsed = started.elapsed();
        let response = declare(
            &incoming_sender,
            0,
            sample as u32,
            coinbase,
            mempool_wtxids.clone(),
            vec![],
        )
        .await;
        let end_to_end_elapsed = started.elapsed();
        assert!(
            matches!(response, JdResponse::Success { .. }),
            "expected Success, got {response:?}"
        );
        let collect_elapsed = end_to_end_elapsed.saturating_sub(sendraw_elapsed);
        release(&incoming_sender, 0, sample as u32).await;

        sendraw_samples.push(sendraw_elapsed);
        collect_samples.push(collect_elapsed);
        end_to_end_samples.push(end_to_end_elapsed);
        println!(
            "{sample}\t{:.3}\t{:.3}\t{:.3}",
            duration_ms(sendraw_elapsed),
            duration_ms(collect_elapsed),
            duration_ms(end_to_end_elapsed)
        );
    }
    print_summary("sendrawtransaction RPC", &mut sendraw_samples);
    print_summary("collectTxs + makeTemplate", &mut collect_samples);
    print_summary("mempool-hit end-to-end", &mut end_to_end_samples);

    println!("\nTxCollection missing-transaction path ({sample_count} samples)");
    println!("sample\tdetect_missing_ms\tadd_make_ms\tcombined_ms");
    let mut detect_samples = Vec::with_capacity(sample_count);
    let mut add_samples = Vec::with_capacity(sample_count);
    let mut combined_samples = Vec::with_capacity(sample_count);
    for (index, transaction) in missing_transactions.into_iter().enumerate() {
        let sample = index + 1;
        let wtxid = transaction.compute_wtxid();
        let request_id = (sample_count + sample) as u32;
        let coinbase = build_valid_coinbase_tx(next_height, &[wtxid]);

        let combined_started = Instant::now();
        let detect_started = Instant::now();
        let response = declare(
            &incoming_sender,
            1,
            request_id,
            coinbase.clone(),
            vec![wtxid],
            vec![],
        )
        .await;
        let detect_elapsed = detect_started.elapsed();
        match response {
            JdResponse::MissingTransactions { missing_wtxids, .. } => {
                assert_eq!(missing_wtxids, vec![wtxid]);
            }
            response => panic!("expected MissingTransactions, got {response:?}"),
        }

        let add_started = Instant::now();
        let response = declare(
            &incoming_sender,
            1,
            request_id,
            coinbase,
            vec![wtxid],
            vec![transaction],
        )
        .await;
        let add_elapsed = add_started.elapsed();
        let combined_elapsed = combined_started.elapsed();
        assert!(
            matches!(response, JdResponse::Success { .. }),
            "expected Success, got {response:?}"
        );
        release(&incoming_sender, 1, request_id).await;

        detect_samples.push(detect_elapsed);
        add_samples.push(add_elapsed);
        combined_samples.push(combined_elapsed);
        println!(
            "{sample}\t{:.3}\t{:.3}\t{:.3}",
            duration_ms(detect_elapsed),
            duration_ms(add_elapsed),
            duration_ms(combined_elapsed)
        );
    }
    print_summary("unknownTxPos", &mut detect_samples);
    print_summary("addMissingTxs + makeTemplate", &mut add_samples);
    print_summary("missing path combined", &mut combined_samples);

    cancellation_token.cancel();
    jdp_thread.join().expect("JDP thread join failed");
}

async fn declare(
    sender: &Sender<JdRequest>,
    downstream_id: usize,
    request_id: u32,
    coinbase_tx: Transaction,
    wtxid_list: Vec<Wtxid>,
    missing_txs: Vec<Transaction>,
) -> JdResponse {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    sender
        .send(JdRequest::DeclareMiningJob {
            downstream_id,
            request_id,
            version: BlockVersion::from_consensus(0x2000_0000),
            coinbase_tx,
            wtxid_list,
            missing_txs,
            response_tx,
        })
        .await
        .expect("failed to send declaration");
    tokio::time::timeout(Duration::from_secs(20), response_rx)
        .await
        .expect("timed out waiting for declaration")
        .expect("declaration response channel dropped")
}

async fn release(sender: &Sender<JdRequest>, downstream_id: usize, request_id: u32) {
    sender
        .send(JdRequest::ReleaseDeclaredJob {
            downstream_id,
            request_id,
        })
        .await
        .expect("failed to release declared job");
}

fn print_summary(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let mean = samples.iter().copied().map(duration_ms).sum::<f64>() / samples.len() as f64;
    println!(
        "{label}: mean={mean:.3} ms, p50={:.3} ms, p95={:.3} ms, p99={:.3} ms, min={:.3} ms, max={:.3} ms",
        duration_ms(percentile(samples, 50)),
        duration_ms(percentile(samples, 95)),
        duration_ms(percentile(samples, 99)),
        duration_ms(samples[0]),
        duration_ms(samples[samples.len() - 1])
    );
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    samples[(samples.len() * percentile).div_ceil(100).saturating_sub(1)]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn coinbase_script_sig_for_height(height: u32) -> ScriptBuf {
    let mut encoded = Vec::new();
    let mut value = height;
    while value > 0 {
        encoded.push((value & 0xff) as u8);
        value >>= 8;
    }
    if encoded.last().is_some_and(|byte| byte & 0x80 != 0) {
        encoded.push(0);
    }
    let mut script = Vec::with_capacity(encoded.len() + 1);
    script.push(encoded.len() as u8);
    script.extend_from_slice(&encoded);
    ScriptBuf::from_bytes(script)
}

fn build_valid_coinbase_tx(next_height: u32, wtxids: &[Wtxid]) -> Transaction {
    let witness_hashes = std::iter::once(Wtxid::all_zeros().to_raw_hash())
        .chain(wtxids.iter().map(|wtxid| wtxid.to_raw_hash()));
    let witness_root: WitnessMerkleNode =
        stratum_apps::stratum_core::bitcoin::merkle_tree::calculate_root(witness_hashes)
            .expect("witness tree has a root")
            .into();
    let reserved = [0_u8; 32];
    let commitment = Block::compute_witness_commitment(&witness_root, &reserved);
    let mut commitment_script = vec![0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    commitment_script.extend_from_slice(&commitment.to_byte_array());

    Transaction {
        version: TxVersion::TWO,
        lock_time: LockTime::ZERO,
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            script_sig: coinbase_script_sig_for_height(next_height),
            sequence: Sequence::MAX,
            witness: Witness::from_slice(&[reserved]),
        }],
        output: vec![
            TxOut {
                value: Amount::ZERO,
                script_pubkey: ScriptBuf::new(),
            },
            TxOut {
                value: Amount::ZERO,
                script_pubkey: ScriptBuf::from_bytes(commitment_script),
            },
        ],
    }
}
