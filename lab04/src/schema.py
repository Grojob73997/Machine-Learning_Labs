"""Fixed Adult data dictionary: never infer allowed categories from test data."""
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORIES = {
    "workclass": "Private|Self-emp-not-inc|Self-emp-inc|Federal-gov|Local-gov|State-gov|Without-pay|Never-worked",
    "education": "Bachelors|Some-college|11th|HS-grad|Prof-school|Assoc-acdm|Assoc-voc|9th|7th-8th|12th|Masters|1st-4th|10th|Doctorate|5th-6th|Preschool",
    "marital-status": "Married-civ-spouse|Divorced|Never-married|Separated|Widowed|Married-spouse-absent|Married-AF-spouse",
    "occupation": "Tech-support|Craft-repair|Other-service|Sales|Exec-managerial|Prof-specialty|Handlers-cleaners|Machine-op-inspct|Adm-clerical|Farming-fishing|Transport-moving|Priv-house-serv|Protective-serv|Armed-Forces",
    "relationship": "Wife|Own-child|Husband|Not-in-family|Other-relative|Unmarried",
    "race": "White|Asian-Pac-Islander|Amer-Indian-Eskimo|Other|Black",
    "sex": "Female|Male",
    "native-country": "United-States|Cambodia|England|Puerto-Rico|Canada|Germany|Outlying-US(Guam-USVI-etc)|India|Japan|Greece|South|China|Cuba|Iran|Honduras|Philippines|Italy|Poland|Jamaica|Vietnam|Mexico|Portugal|Ireland|France|Dominican-Republic|Laos|Ecuador|Taiwan|Haiti|Columbia|Hungary|Guatemala|Nicaragua|Scotland|Thailand|Yugoslavia|El-Salvador|Trinadad&Tobago|Peru|Hong|Holand-Netherlands",
    "income": "<=50K|>50K",
}
CATEGORIES = {k: v.split("|") + (["?"] if k != "income" else []) for k, v in CATEGORIES.items()}
CATEGORICAL = [c for c in CATEGORIES if c != "income"]
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
# Extra dictionary checks are identified separately from the manual's four ranges.
EXTRA_RANGES = {"education-num": (1, 16), "fnlwgt": (2, None)}
EDUCATION_CODES = dict(zip(
    ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
     "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"],
    range(1, 17)))
