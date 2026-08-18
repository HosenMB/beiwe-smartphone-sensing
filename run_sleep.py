import sleepmetric as sm

# gp83emoi = your data -> one summary row per night (Maryland time, labelled by evening date)
df = sm.analyze_beiwe("Beiwe_Data/gp83emoi/accelerometer",
                      save_preprocessed="Output/gp83emoi_clean.csv")
df.to_csv("Output/gp83emoi_sleep_metrics.csv", index=False)
print(df.to_string(index=False))
