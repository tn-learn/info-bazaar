import argparse
import pickle

from bazaar.dataset_builder import (
    load_or_parse_arxiv_data,
    load_or_scrape_openalex_works,
    download_arxiv_papers,
    load_or_parse_latex_source,
    filter_and_load_oa_works,
    build_blocks,
    extract_questions_from_blocks,
)


def main():
    parser = argparse.ArgumentParser(description="Process arXiv papers")
    parser.add_argument("--category", choices=["machine-learning", "astrophysics", "llm"], help="The category to process")
    parser.add_argument("--model_name", default="RemoteLlama-2-70b-chat-hf", help="The model to use")
    parser.add_argument("--data_root", default="/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data", help="data root")
    args = parser.parse_args()

    data_root = args.data_root
    category = args.category
    model_name = args.model_name

    metadata = load_or_parse_arxiv_data(category, data_root)
    oa_works = load_or_scrape_openalex_works(category, metadata, data_root)
    download_arxiv_papers(category, oa_works, data_root)
    papers = load_or_parse_latex_source(category, data_root)
    oa_works_w_arxiv, paper_samples = filter_and_load_oa_works(category, papers, oa_works, data_root=data_root)
    dataset_step_0 = build_blocks(category, oa_works_w_arxiv, data_root, paper_samples)
    dataset_step_1 = extract_questions_from_blocks(category, dataset_step_0, model_name=model_name)

    # Save the final dataset as needed
    pickle.dump(dataset_step_1, open(f"data/{category}/final_dataset.pkl", "wb"))


if __name__ == "__main__":
    main()
