from graph_experiment.dataset import prepare_graph_dataset


if __name__ == "__main__":
    metadata = prepare_graph_dataset()
    print("Graph experiment dataset prepared")
    print(metadata["chronological_split"])