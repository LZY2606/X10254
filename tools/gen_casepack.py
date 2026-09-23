"""重新生成 casepack/*.json（确定性，numpy 种子固定）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import casepack


def dump(name, data):
    path = os.path.join(casepack.BASE_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote", path)


def main():
    baseline = casepack.build_baseline()
    dump("baseline.json", baseline)
    dump("revisions.json", casepack.build_revisions(baseline))
    dump("counterpoints.json", casepack.build_counterpoints())


if __name__ == "__main__":
    main()
