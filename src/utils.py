from pandas import DataFrame

def data2excel(name_excel:str, loss:list, metrics:dict):
    df = DataFrame({
        "eposh": [i for i in range(1, len(loss)+1)],
        "loss": loss,
        "test_loss": metrics["Test_loss"],
        "HR@10": metrics["HR@10"],
        "Precision@10": metrics["Precision@10"],
        "Recall@10": metrics["Recall@10"],
        "MAP@10": metrics["MAP@10"],
        "NDCG@10": metrics["NDCG@10"],
        "MRR@10": metrics["MRR@10"],
        
    })
    
    df.to_excel(f"{name_excel}.xlsx")
    
    return df