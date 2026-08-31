"""
==============================================================================
Pure-Python GeoIP & Real BGP ASN Resolution Engine (BitSentinel-AI - Fix 2 & 3b)
==============================================================================
Enriches synthetic Bitcoin network metadata with REAL GeoIP & BGP ASN telemetry:
  1. Uses GeoIP2Fast for ISO Country code resolution.
  2. Uses pure-Python binary-search BGP table (from iptoasn/RouteViews archive)
     for genuine ASN number & organization resolution (Zero C++ compiler required).
  3. Strict diagnostic self-test on 8.8.8.8 (AS15169 / Google) and 1.1.1.1 (AS13335 / Cloudflare).
  4. Resolves src_ip and dst_ip across all 46,564 rows in processed/network_metadata.csv.
  5. Preserves is_injected_pattern and is_legit_bursty_cluster flags.
  6. Exports enriched dataset to processed/network_metadata_geo.csv.
==============================================================================
"""

import os
import sys
import gzip
import bisect
import socket
import struct
import urllib.request
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("processed")
INPUT_CSV = PROCESSED_DIR / "network_metadata.csv"
RAW_DIR = Path("raw data")
ASN_TSV_GZ = RAW_DIR / "ip2asn-v4.tsv.gz" if (RAW_DIR / "ip2asn-v4.tsv.gz").exists() else Path("ip2asn-v4.tsv.gz")

SEP = "=" * 80

print(f"\n{SEP}")
print("  BITCOIN NETWORK METADATA - PURE-PYTHON GEOIP & REAL ASN RESOLUTION")
print(SEP)

# ------------------------------------------------------------------------------
# 1. Download & Index Real BGP IP-to-ASN Database (Pure Python)
# ------------------------------------------------------------------------------
print("\n[1] Initializing pure-Python BGP IP-to-ASN database ...")

def download_with_headers(url, dest_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())

if not ASN_TSV_GZ.exists() or ASN_TSV_GZ.stat().st_size < 1000000:
    print("    Downloading latest BGP IP-to-ASN database (ip2asn-v4.tsv.gz, ~11 MB) ...")
    asn_urls = [
        "https://iptoasn.com/data/ip2asn-v4.tsv.gz",
        "https://cdn.jsdelivr.net/gh/sapics/ip-location-db@master/asn/asn-ipv4.tsv.gz"
    ]
    download_ok = False
    for url in asn_urls:
        try:
            print(f"    Fetching from: {url} ...")
            download_with_headers(url, ASN_TSV_GZ)
            if ASN_TSV_GZ.exists() and ASN_TSV_GZ.stat().st_size > 1000000:
                download_ok = True
                print(f"    ✓ Download complete ({ASN_TSV_GZ.stat().st_size / 1024 / 1024:.2f} MB).")
                break
        except Exception as e:
            print(f"    Download failed from {url}: {e}")

    if not download_ok:
        print("\n[ERROR] Could not automatically download ip2asn-v4.tsv.gz.")
        sys.exit(1)
else:
    print(f"    ✓ Using cached BGP IP-to-ASN database ({ASN_TSV_GZ.stat().st_size / 1024 / 1024:.2f} MB).")

def ip_to_int(ip_str):
    try:
        return struct.unpack("!I", socket.inet_aton(ip_str))[0]
    except Exception:
        return None

# Load and index ranges in memory for binary search
print("    Indexing BGP routing table in memory ...")
start_ips = []
end_ips = []
asn_numbers = []
asn_countries = []
asn_org_names = []

with gzip.open(ASN_TSV_GZ, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            s_ip = ip_to_int(parts[0])
            e_ip = ip_to_int(parts[1])
            if s_ip is not None and e_ip is not None:
                start_ips.append(s_ip)
                end_ips.append(e_ip)
                asn_numbers.append(f"AS{parts[2]}" if not parts[2].startswith("AS") else parts[2])
                asn_countries.append(parts[3].upper())
                asn_org_names.append(parts[4])

print(f"    ✓ Loaded {len(start_ips):,} BGP routing prefixes into memory.")

def lookup_real_asn(ip_str):
    ip_int = ip_to_int(ip_str)
    if ip_int is None:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    
    idx = bisect.bisect_right(start_ips, ip_int) - 1
    if idx >= 0 and ip_int <= end_ips[idx]:
        c_code = asn_countries[idx] if asn_countries[idx] not in ("-", "", "NONE") else "UNKNOWN"
        a_num  = asn_numbers[idx] if asn_numbers[idx] not in ("AS0", "AS-", "ASNONE") else "UNKNOWN"
        a_org  = asn_org_names[idx] if asn_org_names[idx] not in ("-", "", "NONE") else "UNKNOWN"
        return c_code, a_num, a_org
    return "UNKNOWN", "UNKNOWN", "UNKNOWN"

# ------------------------------------------------------------------------------
# 2. Strict Diagnostic Self-Test (8.8.8.8 -> Google, 1.1.1.1 -> Cloudflare)
# ------------------------------------------------------------------------------
print("\n[2] Running strict diagnostic self-test against ground-truth public IPs ...")

c_8888, asn_8888, org_8888 = lookup_real_asn("8.8.8.8")
c_1111, asn_1111, org_1111 = lookup_real_asn("1.1.1.1")

print(f"    Lookup 8.8.8.8 -> Country: {c_8888:>2} | Real BGP ASN: {asn_8888:<8} | Org: '{org_8888}'")
print(f"    Lookup 1.1.1.1 -> Country: {c_1111:>2} | Real BGP ASN: {asn_1111:<8} | Org: '{org_1111}'")

is_8888_ok = ("15169" in asn_8888) and ("GOOGLE" in org_8888.upper())
is_1111_ok = ("13335" in asn_1111) and ("CLOUDFLARE" in org_1111.upper())

if not (is_8888_ok and is_1111_ok):
    print("\n[CRITICAL ERROR] ASN self-test failed! Aborting.")
    sys.exit(1)

print("    ✓ Diagnostic self-test PASSED.")

# ------------------------------------------------------------------------------
# 3. Load Fix 1 / 3b Network Metadata Dataset
# ------------------------------------------------------------------------------
print(f"\n[3] Loading network metadata from {INPUT_CSV} ...")

if not INPUT_CSV.exists():
    print(f"\n[ERROR] Missing {INPUT_CSV}! Please run 'python generate_network_metadata.py' first.\n")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
total_rows = len(df)
print(f"    Loaded {total_rows:,} total transaction records.")

if "datetime_utc" not in df.columns and "timestamp" in df.columns:
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s").astype(str)

if "is_legit_bursty_cluster" not in df.columns:
    df["is_legit_bursty_cluster"] = False

# ------------------------------------------------------------------------------
# 4. Perform Fast Cached Resolution
# ------------------------------------------------------------------------------
print("\n[4] Performing genuine GeoIP + BGP ASN resolution across unique IPv4 addresses ...")

all_unique_ips = pd.concat([df["src_ip"], df["dst_ip"]]).dropna().unique()
print(f"    Total unique IP addresses to resolve: {len(all_unique_ips):,}")

ip_cache = {}
for ip in all_unique_ips:
    ip_cache[ip] = lookup_real_asn(ip)

src_results = [ip_cache.get(ip, ("UNKNOWN", "UNKNOWN", "UNKNOWN")) for ip in df["src_ip"]]
dst_results = [ip_cache.get(ip, ("UNKNOWN", "UNKNOWN", "UNKNOWN")) for ip in df["dst_ip"]]

df["src_country"]   = [r[0] for r in src_results]
df["src_asn"]       = [r[1] for r in src_results]
df["src_asn_name"]  = [r[2] for r in src_results]

df["dst_country"]   = [r[0] for r in dst_results]
df["dst_asn"]       = [r[1] for r in dst_results]
df["dst_asn_name"]  = [r[2] for r in dst_results]

df["same_asn_as_cluster_peers"] = False

# ------------------------------------------------------------------------------
# 5. Save Enriched Dataset
# ------------------------------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[5] Saved Enriched Dataset -> {OUTPUT_CSV}")

print(f"\n{SEP}")
print("  GENUINE GEOIP & BGP ASN ENRICHMENT COMPLETE")
print(SEP)
