"""Local HTTP/TAR tests: no public network or large datasets are used."""
from __future__ import annotations

import hashlib
import io
import ssl
import tarfile
import tempfile
import threading
import unittest
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from src.download import DownloadError, download_file, extract_selected


@contextmanager
def archive_server(body: bytes, range_support=True, bad_range=False,
                   interrupt_once=False, transient_once=False,
                   bad_content_length=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            requests.append(header)
            if transient_once and len(requests) == 1:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if header and range_support:
                start = int(header.removeprefix("bytes=").removesuffix("-"))
                if start >= len(body):
                    self.send_response(416)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206)
                reported_start = start + 1 if bad_range else start
                self.send_header("Content-Range", f"bytes {reported_start}-{len(body)-1}/{len(body)}")
                payload = body[start:]
            else:
                self.send_response(200)
                payload = body
            self.send_header("Content-Length", str(len(payload) + int(bad_content_length)))
            self.end_headers()
            if interrupt_once and len(requests) == 1:
                self.wfile.write(payload[:1024 * 1024 + 31])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/archive.tar", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.payload = b"small local archive body\x00" * 17
        self.md5 = hashlib.md5(self.payload).hexdigest()
        self.destination = self.root / "archive.tar"

    def test_resume_206_and_verified_cache(self):
        partial = self.destination.with_name("archive.tar.part")
        partial.write_bytes(self.payload[:41])
        with archive_server(self.payload) as (url, requests):
            result = download_file(url, self.destination, self.md5, len(self.payload), retries=0)
            self.assertEqual(requests, ["bytes=41-"])
            self.assertEqual(self.destination.read_bytes(), self.payload)
            self.assertFalse(partial.exists())
            self.assertEqual(result["status"], "downloaded")
            cached = download_file(url, self.destination, self.md5, len(self.payload), retries=0)
            self.assertEqual(cached["status"], "cached")
            self.assertEqual(len(requests), 1)

    def test_ignored_range_restarts_instead_of_appending(self):
        self.destination.with_name("archive.tar.part").write_bytes(self.payload[:31])
        with archive_server(self.payload, range_support=False) as (url, requests):
            download_file(url, self.destination, self.md5, len(self.payload), retries=0)
        self.assertEqual(requests, ["bytes=31-"])
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_md5_failure_never_publishes(self):
        with archive_server(self.payload) as (url, _):
            with self.assertRaisesRegex(DownloadError, "MD5 mismatch"):
                download_file(url, self.destination, "0" * 32, len(self.payload), retries=0)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_name("archive.tar.part").exists())

    def test_bad_existing_destination_is_not_returned_as_cached(self):
        self.destination.write_bytes(b"corrupt old archive")
        with archive_server(self.payload) as (url, _):
            result = download_file(url, self.destination, self.md5, len(self.payload), retries=0)
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_wrong_range_is_rejected_before_append(self):
        partial = self.destination.with_name("archive.tar.part")
        original = self.payload[:21]
        partial.write_bytes(original)
        with archive_server(self.payload, bad_range=True) as (url, _):
            with self.assertRaisesRegex(DownloadError, "Content-Range"):
                download_file(url, self.destination, self.md5, len(self.payload), retries=0)
        self.assertEqual(partial.read_bytes(), original)
        self.assertFalse(self.destination.exists())

    def test_complete_partial_with_unknown_size_handles_416(self):
        self.destination.with_name("archive.tar.part").write_bytes(self.payload)
        with archive_server(self.payload) as (url, _):
            result = download_file(url, self.destination, self.md5, retries=0)
        self.assertEqual(result["bytes"], len(self.payload))
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_interrupted_transfer_retries_from_persisted_offset(self):
        payload = b"r" * (2 * 1024 * 1024 + 100)
        checksum = hashlib.md5(payload).hexdigest()
        with archive_server(payload, interrupt_once=True) as (url, requests):
            with mock.patch("src.download.time.sleep") as sleep:
                result = download_file(url, self.destination, checksum, len(payload), retries=1)
        self.assertEqual(requests, [None, f"bytes={1024 * 1024 + 31}-"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(self.destination.read_bytes(), payload)
        sleep.assert_called_once_with(1)

    def test_transient_http_error_retries(self):
        with archive_server(self.payload, transient_once=True) as (url, requests):
            with mock.patch("src.download.time.sleep"):
                result = download_file(url, self.destination, self.md5, len(self.payload), retries=1)
        self.assertEqual(requests, [None, None])
        self.assertEqual(result["attempts"], 2)

    def test_range_content_length_mismatch_is_rejected(self):
        partial = self.destination.with_name("archive.tar.part")
        original = self.payload[:23]
        partial.write_bytes(original)
        with archive_server(self.payload, bad_content_length=True) as (url, _):
            with self.assertRaisesRegex(DownloadError, "Content-Length"):
                download_file(url, self.destination, self.md5, len(self.payload), retries=0)
        self.assertEqual(partial.read_bytes(), original)

    def test_timeout_then_certificate_failure_preserves_resume_offset(self):
        partial = self.destination.with_name("archive.tar.part")
        original = self.payload[:41]
        partial.write_bytes(original)
        certificate_error = urllib.error.URLError(ssl.SSLCertVerificationError(
            1, "self-signed certificate in certificate chain"))
        with mock.patch("src.download.urllib.request.urlopen",
                        side_effect=[TimeoutError("read timed out"), certificate_error]) as open_url, \
                mock.patch("src.download.time.sleep") as sleep:
            with self.assertRaisesRegex(DownloadError, "TLS certificate verification failed") as caught:
                download_file("https://example.test/archive.tar", self.destination,
                              self.md5, len(self.payload), retries=4)
        self.assertEqual(open_url.call_count, 2)
        self.assertTrue(all(call.args[0].get_header("Range") == "bytes=41-"
                            for call in open_url.call_args_list))
        sleep.assert_called_once_with(1)
        self.assertIn("--ca-bundle", str(caught.exception))
        self.assertIn(str(partial), str(caught.exception))
        self.assertIn("41 bytes", str(caught.exception))
        self.assertEqual(partial.read_bytes(), original)
        self.assertFalse(self.destination.exists())

    def test_direct_certificate_error_is_reported_without_retry(self):
        with mock.patch("src.download.urllib.request.urlopen",
                        side_effect=ssl.SSLCertVerificationError(1, "untrusted issuer")) as open_url, \
                mock.patch("src.download.time.sleep") as sleep:
            with self.assertRaisesRegex(DownloadError, "SSL_CERT_FILE"):
                download_file("https://example.test/archive.tar", self.destination, self.md5)
        open_url.assert_called_once()
        sleep.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_custom_ca_context_is_verified_and_reused_when_resuming(self):
        partial = self.destination.with_name("archive.tar.part")
        partial.write_bytes(self.payload[:41])
        response = io.BytesIO(self.payload[41:])
        response.status = 206
        response.headers = {"Content-Length": str(len(self.payload) - 41),
                            "Content-Range": f"bytes 41-{len(self.payload) - 1}/{len(self.payload)}"}
        context = ssl.create_default_context()
        ca_path = self.root / "institution-ca.pem"
        with mock.patch("src.download.ssl.create_default_context", return_value=context), \
                mock.patch.object(context, "load_verify_locations") as load_ca, \
                mock.patch("src.download.urllib.request.urlopen",
                           side_effect=[TimeoutError("read timed out"), response]) as open_url, \
                mock.patch("src.download.time.sleep"):
            result = download_file("https://example.test/archive.tar", self.destination,
                                   self.md5, len(self.payload), retries=1, ca_bundle=ca_path)
        load_ca.assert_called_once_with(cafile=str(ca_path))
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertTrue(all(call.kwargs["context"] is context for call in open_url.call_args_list))
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertFalse(partial.exists())

    def test_invalid_ca_bundle_preserves_partial_without_request(self):
        for content in (None, "not a PEM certificate"):
            with self.subTest(content=content):
                ca_path = self.root / "bad-ca.pem"
                if content is not None:
                    ca_path.write_text(content)
                partial = self.destination.with_name("archive.tar.part")
                original = self.payload[:41]
                partial.write_bytes(original)
                with mock.patch("src.download.urllib.request.urlopen") as open_url:
                    with self.assertRaisesRegex(DownloadError, "Cannot load trusted CA bundle"):
                        download_file("https://example.test/archive.tar", self.destination,
                                      self.md5, ca_bundle=ca_path)
                open_url.assert_not_called()
                self.assertEqual(partial.read_bytes(), original)
                self.assertFalse(self.destination.exists())


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.archive = self.root / "archive.tar"
        self.destination = self.root / "selected"

    def create_tar(self, entries):
        with tarfile.open(self.archive, "w") as bundle:
            for name, content, kind in entries:
                member = tarfile.TarInfo(name)
                if kind == "file":
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
                else:
                    member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                    member.linkname = "../outside"
                    bundle.addfile(member)

    def test_filter_normalize_and_rewrite_existing_file(self):
        selected = "wrapper/another/year_month=22-02/plugin=ipmi_pub/metric=total_power/node=1/part.parquet"
        self.create_tar([(selected, b"valid-data", "file"),
                         ("wrapper/year_month=22-02/plugin=weather/metric=temp/part.parquet", b"weather", "file"),
                         ("wrapper/year_month=22-02/plugin=ipmi_pub/metric=ambient/part.parquet", b"other", "file"),
                         ("readme.txt", b"not selected", "file")])
        target = self.destination / "year_month=22-02/plugin=ipmi_pub/metric=total_power/node=1/part.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"broken-dat")  # Same size is deliberately not considered valid.
        report = extract_selected(self.archive, self.destination,
                                  {("ipmi", "total_power"), ("weather_pub", "temp")})
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["bytes"], len(b"valid-dataweather"))
        self.assertEqual(report["missing_metrics"], [])
        self.assertEqual(target.read_bytes(), b"valid-data")
        self.assertFalse((self.destination / "wrapper").exists())
        self.assertFalse((self.destination / "year_month=22-02/plugin=ipmi_pub/metric=ambient").exists())

    def test_traversal_absolute_and_windows_drive_paths_are_rejected(self):
        for name in ("../outside.parquet", "/absolute.parquet", "C:/drive.parquet",
                     "wrapper\\..\\outside.parquet"):
            with self.subTest(name=name):
                self.create_tar([(name, b"x", "file")])
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    extract_selected(self.archive, self.destination, {("ipmi", "total_power")})

    def test_all_links_are_rejected(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                self.create_tar([("unselected/link", b"", kind)])
                with self.assertRaisesRegex(ValueError, "links"):
                    extract_selected(self.archive, self.destination, {("ipmi", "total_power")})


if __name__ == "__main__":
    unittest.main()
