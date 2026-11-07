import argparse
import json

from vacv.data import check_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate format, finite values and exact sample order; not prediction quality")
    parser.add_argument("submission")
    parser.add_argument("sample")
    args = parser.parse_args()
    print(json.dumps(check_submission(args.submission, args.sample), indent=2))
