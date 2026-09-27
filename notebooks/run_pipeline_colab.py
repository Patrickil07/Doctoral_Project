# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # The Pipeline Paradox — CPS / O\*NET pipeline runner (Colab)
#
# This notebook contains **no analysis code**. It pulls the versioned scripts
# from GitHub and runs `src/00`–`src/07` in order, so every number can be
# traced to a commit.
#
# * **Code** comes from GitHub (this repo).
# * **Data** stays on Google Drive and is linked into `data/`, so large
#   downloads (IPUMS, USAJOBS) survive between Colab sessions and never enter git.
#
# One-time setup in Colab's 🔑 **Secrets** panel (toggle *Notebook access* on):
#
# | Secret | Used by | Notes |
# |---|---|---|
# | `GITHUB_TOKEN` | clone | fine-grained token, read-only *Contents* on this repo |
# | `IPUMS_API_KEY` | step 04 | https://account.ipums.org/api_keys (register for **CPS**) |
# | `BLS_CONTACT_EMAIL` | step 00 | BLS requires a contact address in requests |

# %%
# 0. Configuration: edit these two paths only
REPO = "patrickil07/doctoral_project"
BRANCH = "main"
DRIVE_DATA = "/content/drive/MyDrive/doctoral project/data/pp/data"   # holds raw/ interim/ out/

# %%
# 1. Mount Drive, fetch the code, link the data folder
import os, pathlib, subprocess
from google.colab import drive, userdata

drive.mount("/content/drive")
pathlib.Path(DRIVE_DATA).mkdir(parents=True, exist_ok=True)

code = pathlib.Path("/content/doctoral_project")
token = userdata.get("GITHUB_TOKEN")
if not code.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH,
                    f"https://x-access-token:{token}@github.com/{REPO}.git", str(code)],
                   check=True)
else:
    subprocess.run(["git", "-C", str(code), "pull", "--ff-only"], check=True)
os.chdir(code)

data = code / "data"
if data.is_symlink():
    data.unlink()
elif data.exists():
    # the repo ships only data/README.md; keep a copy next to the Drive data
    subprocess.run(["cp", "-n", "data/README.md", f"{DRIVE_DATA}/README.md"], check=False)
    subprocess.run(["rm", "-rf", "data"], check=True)
data.symlink_to(DRIVE_DATA)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print(f"code: {REPO}@{commit}  |  data: {DRIVE_DATA}")
print("Record this commit next to any table you copy into a chapter.")

# %%
# 2. Dependencies and secrets
# %pip install -q -r requirements.txt
for name in ("IPUMS_API_KEY", "BLS_CONTACT_EMAIL"):
    try:
        os.environ[name] = userdata.get(name)
        print(f"✅ {name} loaded")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  {name} not available ({type(e).__name__})")

# %% [markdown]
# ## Data acquisition

# %%
# 00  OEWS national, Census 2018 crosswalk, CPI-U  (exposure_soc.csv is built by hand)
# !python src/00_fetch_public_inputs.py --oews-year 2019

# %%
# 01  O*NET release pinned for the main specification
# !python src/01_download_onet.py --release 30.3

# %%
# 04  IPUMS CPS extract (skips if already downloaded; --force to re-request)
# !python src/04_ipums_extract.py --start 2019-01 --end 2025-12

# %%
# 06  USAJOBS historic announcements (resumable; re-run after interruptions)
# !python src/06_usajobs_historic.py --start 2019 --end 2025

# %%
# 06  completion report only
# !python src/06_usajobs_historic.py --status

# %% [markdown]
# ## Measures, sample, estimation

# %%
# 02  occupation-level task composition + ILR balances
# !python src/02_build_task_composition.py --onet data/raw/onet_30_3

# %%
# 03  crosswalk to CPS occupation codes, OEWS employment weights
# !python src/03_crosswalk.py

# %%
# 05  estimation sample (Table 4.1 construction log is printed)
# !python src/05_build_sample.py

# %%
# 07  RQ1–RQ4
# !python src/07_estimate.py

# %%
# 07  robustness: drop the pandemic window
# !python src/07_estimate.py --drop-pandemic --out data/out/results_drop_pandemic
