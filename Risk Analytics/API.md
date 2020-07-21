Input: 

```
/predict?Sex=male&Housing=free&Savings accounts=little&Checking account=little&Purpose=business&Job=unskilled and non-resident&Age=23&Duration in Months=32&Credit amount=2345
```

Output: 

```json
{
"bad": 0.0, 
"good": 1.0
}

```

sample
```json
```json
{
    "factors": [
        {
            "factor_name": "fact_1",
            "Range": [
                20,
                30
            ],
            "Interval": 1,
            "Type": "categorical"
        },
        {
            "factor_name": "fact_1",
            "Range": [
                20,
                30
            ],
            "Interval": 0.1,
            "Type": "continuous"
        },
        {
            "factor_name": "fact_3",
            "Range": [
                "value_1",
                "value_2"
            ],
            "Interval": "None",
            "Type": "categorical"
        }
    ]
}
```