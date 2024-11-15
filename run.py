from workflow.crawling import Crawling_for_naver
from jobflow import Flow,run_locally
import argparse



def main():
    args = build_default_arg_parser().parse_args()
    run(args)


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_path",type=str,required=True)
    parser.add_argument("--category",type=str,required=True)
    parser.add_argument("--choice_type",type=str,required=True)
    parser.add_argument("--product_type",type=str,required=True)

    parser.add_argument("--product_threshold",type=int)
    parser.add_argument("--n_pages",type=int)

    return parser


def run(args: argparse.Namespace) -> None:

    crawl = Crawling_for_naver(path = args.data_path,category=args.category,choice_type=args.choice_type,product_type=args.product_type,threshold=args.product_threshold,n_pages=args.n_pages)
    job1 = crawl.get_url()
    job2 = crawl.Crawling(product_dict=job1.output,output_filename=f'{args.product_type}.txt')

    flow = Flow([job1,job2])
    run_locally(flow)


if __name__ == "__main__":
    main()

