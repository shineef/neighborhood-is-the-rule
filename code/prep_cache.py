"""Download the 25 public games into ./environment_files and fill pt_cache with every
(game, moves, seed) that the experiments read:
  (3000, 11)  pretraining pool                 exp_meta.py / exp_pretrain.py / exp_rollout_ft.py
  (900, 23)   held-out 300 train + 600 test    same scripts + exp_lookup.py
  (3600, 31)  data curve of the tables         exp_lookup.py
  unbroken runs (4000, 99, at least 11 moves)  exp_rollout_ft.py
Run from the code directory: python prep_cache.py download, then python prep_cache.py fill [n_procs]
"""
import os, sys, time
from multiprocessing import Pool


def download():
    from arc_agi import Arcade
    arc = Arcade()
    ids = sorted({e.game_id for e in arc.get_environments()})
    for g in ids:
        arc.make(g)
    print(f"downloaded {len(ids)} games", flush=True)
    return ids


def fill(gid):
    import wmcache
    t0 = time.time()
    wmcache.collect(gid, 3000, 11)
    wmcache.collect(gid, 900, 23)
    wmcache.collect(gid, 3600, 31)
    ch = wmcache.collect_chains(gid, 4000, 99, 11)
    return f"{gid}: chains={len(ch)} ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    # python prep_cache.py download      (needs internet)
    # python prep_cache.py fill [n]      (offline, fills pt_cache)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "download":
        download()
    else:
        os.environ["OPERATION_MODE"] = "offline"
        import wmcache
        games = wmcache.all_games()
        print(f"{len(games)} games with metadata", flush=True)
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as p:
            for line in p.imap_unordered(fill, games):
                print(line, flush=True)
        print("PREP DONE", flush=True)
