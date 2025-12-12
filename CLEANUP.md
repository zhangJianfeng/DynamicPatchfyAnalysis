CLEANUP: Repository history rewrite
=================================

Date: 2025-12-12

Summary
-------
This repository had large binary files (training checkpoints, tensorboard runs, and a CSV dataset) in its history which made the repo large. I removed those paths from git history and force-pushed a cleaned history to the remote.

What was removed from history
-----------------------------
- `checkpoints/` (many `.pt` files)
- `runs/` (tensorboard event files)
- `ETTh1.csv`

Commands executed (for reproducibility)
--------------------------------------
1. Installed git-filter-repo (local user install):

   pip install --user git-filter-repo

2. Purged paths from history (force):

   git-filter-repo --force --invert-paths --path checkpoints --path runs --path ETTh1.csv

3. Expired reflog and garbage-collected:

   git reflog expire --expire=now --all
   git gc --prune=now --aggressive

4. Re-added remote (git-filter-repo removed it) and force-pushed cleaned history:

   git remote add origin git@github.com:zhangjianfeng/DynamicPatchfyAnalysis.git
   git push --force origin main

Notes and consequences
----------------------
- The above operations rewrite the `main` branch history. Any collaborators or CI that cloned the repository previously will need to re-clone or reset to the new history to avoid conflicts.
- The removed files remain on your local filesystem (they were removed from the git index, not deleted locally), but are no longer present in the git history on the remote.

If you want, I can also:
- create a `release/` tag containing the current cleaned commit, or
- produce a short email/PR template you can send to collaborators explaining they must re-clone.
