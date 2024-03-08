import argparse

import submitit_automatic_label_ram_save_json as main_func

def parse_args():
    main_parser = main_func.get_args_parser()
    parser = argparse.ArgumentParser("Train Data Locally", parents=[main_parser])
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job (ignored in local training)")
    # parser.add_argument("--device", default="cuda", type=str, help="Device to use for training")
    # Add any other arguments you need
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup or modify args as needed for local execution
    # For example, setting up device
    # args.device = "cuda"  # Assuming you have a CUDA-compatible GPU and want to use it
    
    # Directly call the main function of your training script/module
    main_func.main(args)

if __name__ == "__main__":
    main()