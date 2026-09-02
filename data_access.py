import pandas as pd


def load_customers(path="E:\\AI Track\\Training and courses\\DEBI\\Technical\\NBE Final Project\\coordinator_agent\\data\\customers.csv"):
    return pd.read_csv(path)

def get_customer(customer_id, path="E:\\AI Track\\Training and courses\\DEBI\\Technical\\NBE Final Project\\coordinator_agent\\data\\customers.csv"):
    df = load_customers(path)
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

if __name__ == "__main__":
    customer = get_customer("CUSTE2483D")
    print(customer)