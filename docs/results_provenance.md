# Linking chapter results to code

Every number in Chapters 4–5 should be traceable to one commit. The routine:

1. **Run from a clean commit.** `git status` shows nothing uncommitted.
2. **Run the pipeline** (`make estimate`, or the Colab runner). The runner
   prints the commit hash; keep it with the output.
3. **Copy aggregate result tables** (coefficients, standard errors, sample
   counts) into `results/chapter4/` and commit them. Aggregate estimates are
   not IPUMS microdata and may be committed; person-level rows may not.
4. **Tag the commit** that produced them:

   ```bash
   git tag -a ch4-v1 -m "Chapter 4 tables: O*NET 30.3, OEWS 2019, IPUMS extract #NNN"
   git push origin ch4-v1
   ```

5. **Log it** in the table below.
6. **Before submission**, archive the final tag with Zenodo (GitHub
   integration → create a release from the tag) to get a DOI to cite in the
   methodology chapter. The repository can stay private until after the viva;
   Zenodo allows restricted-access records until then.

## Results log

| Tag | Date | Chapter / tables | Data versions | Notes |
|---|---|---|---|---|
| | | | | |
