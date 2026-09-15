"""Assert the golden V3.3 certificate identity on every supported CI host."""

from pathlib import Path
from tempfile import TemporaryDirectory

from dualstream.compact_evidence import VERSION_V33, encode_compact_sequence
from dualstream.verifier import verify_evidence_artifact
from dualstream.work_certificate import certificate_digest


EXPECTED_DIGEST = "8e88bf885de182121bbf9aa217d2a67ba649840fb47feb2f861f79923c9f6945"


def main() -> None:
    rows = [
        {
            "chosen_id": index,
            "topk_ids": [index, index + 1, index + 2],
            "topk_scores": [0.7, 0.2, 0.1],
        }
        for index in range(1024)
    ]
    artifact = encode_compact_sequence(
        rows, wire_version=VERSION_V33, adaptive_k=False
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "portable-certificate.dsae"
        path.write_bytes(artifact)
        report = verify_evidence_artifact(path)

    if not report.ok or report.work_certificate is None:
        raise SystemExit(f"portable verification failed: {report.errors}")
    actual = certificate_digest(report.work_certificate)
    if actual != EXPECTED_DIGEST:
        raise SystemExit(
            f"portable certificate mismatch: expected {EXPECTED_DIGEST}, got {actual}"
        )
    print(actual)


if __name__ == "__main__":
    main()
