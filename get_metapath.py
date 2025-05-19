
def bestMetapaths(model):
    model.eval()

    # Get relation importance scores
    relation_importance = model.get_relation_importance()
    print("Relation Importance:")
    for rel, score in relation_importance.items():
        print(f"{rel}: {score:.4f}")

    # Get top metapaths with valid paths
    print("\nTop Valid Metapaths:")
    for path, score in model.get_top_metapaths(5):
        print(f"{path} | Score: {score:.4f}")