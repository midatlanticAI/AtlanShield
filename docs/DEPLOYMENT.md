# ATLAN Deployment Guide: Low-Cost & Efficient

Because we designed ATLAN to use **Resonant Cognitive Architecture (RCA)** instead of heavy Transformer models, it does **not** require GPUs or massive RAM. It is lightweight, CPU-efficient, and can run on minimal hardware.

## Recommended Hosting: DigitalOcean

### **The "Starter" Architecture**
*   **Droplet Type**: Basic Droplet (Regular SSD)
*   **Size**: **1 GB Memory / 1 CPU**
*   **Cost**: **$6.00 / month**
*   **OS**: Ubuntu 24.04 (LTS) x64

> **Why this works:** Typical RAG/LLM apps need 4GB-16GB RAM for vector stores (Pinecone/Chroma) and models (Torch/HuggingFace). ATLAN uses in-memory python lists and raw math (`utils.py`). It idols at <100MB RAM. The 1GB node provides ample headroom for the OS and the Truth Graph (up to ~100k nodes).

### **Scaling Up (If needed)**
If you ingest massive legal libraries (1M+ lines of text), upgrade to:
*   **Size**: 2 GB Memory / 2 CPUs
*   **Cost**: $18.00 / month
*   **Benefit**: The 2nd CPU core helps handle concurrent requests during load bursts.

---

## Alternative: "Zero Cost" Options
Since you mentioned cost is prohibitive, consider these "Always Free" tiers. ATLAN works perfectly on them:

1.  **Oracle Cloud Free Tier** (Highly Recommended if you can pass signup)
    *   **Specs**: Up to 4 ARM Ampere CPUs, **24 GB RAM**.
    *   **Cost**: **$0.00 / month**.
    *   **Fit**: Overkill for ATLAN, but completely free.
2.  **AWS Free Tier**
    *   **Instance**: `t2.micro` or `t3.micro` (750 hours/month).
    *   **Cost**: **Free for 12 months**.
    *   **Fit**: 1GB RAM is sufficient.
3.  **Google Cloud Free Tier**
    *   **Instance**: `e2-micro`.
    *   **Cost**: **Free**.
    *   **Fit**: Tight on RAM (only 0.6-1GB), but ATLAN can likely squeeze in.

---

## Deployment Instructions (DigitalOcean / Ubuntu)

### 1. Initial Setup
SSH into your droplet:
```bash
ssh root@<your-ip>
apt update && apt upgrade -y
apt install python3-pip python3-venv git -y
```

### 2. Clone & Install
```bash
git clone <your-repo-url> atlan
cd atlan/commercialization
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn requests
# No heavy ML libraries needed!
```

### 3. Run with Systemd (Production)
Create a service file to keep it running:
`nano /etc/systemd/system/atlan.service`

```ini
[Unit]
Description=Atlan Server
After=network.target

[Service]
User=root
WorkingDirectory=/root/atlan/commercialization/atlan_server
ExecStart=/root/atlan/commercialization/venv/bin/uvicorn main:app --host 0.0.0.0 --port 80
Restart=always

[Install]
WantedBy=multi-user.target
```

**Start it:**
```bash
systemctl daemon-reload
systemctl start atlan
systemctl enable atlan
```

### 4. Zero-Downtime Updates
Since the application is stateless (except for `atlan_memory.json` which persists to disk), you can update simply:
```bash
cd atlan
git pull
systemctl restart atlan
```
