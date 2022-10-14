from prepare import text_to_int, clean
from machine_learn import model
import pandas as pd


test = pd.read_csv('C:/Users/n.ushanov/Downloads/insclass_test.csv')
test_ids = test["id"]

test = test.drop(["id"], axis=1)
text_to_int(test)
clean(test)

submission_preds = model.predict(test)

itog = pd.DataFrame({"id": test_ids.values,
                   "target": submission_preds,
                  })

itog.loc[(itog["target"] < 0.5), 'target'] = 0
itog.loc[(itog["target"] >= 0.5), 'target'] = 1

itog.to_csv("C:/Users/n.ushanov/Downloads/itog.csv", index=False, sep=',')
