from scipy.stats import ttest_rel
from scipy.stats import pearsonr

# Score Results collected from the 25 Test set
original_scores = [3,3,5,3,4,5,3,5,4,3,5,4,4,2,5,3,5,4,5,5,5,5,4,5,4,5]

bert_scores = [4,4,4,4,3,4,3,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

t5_scores = [4,3,3,3,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

mistral_scores= [4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]


# === Paired t-test results===
t_stat, p_value = ttest_rel(original_scores, bert_scores)

print("Paired t-test Results for Bert-Base:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

t_stat, p_value = ttest_rel(original_scores, t5_scores)

print("Paired t-test Results for t5-Base:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

t_stat, p_value = ttest_rel(original_scores, mistral_scores)

print("Paired t-test Results for mistral:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

# === Run Pearson correlation ===
corr, p_value = pearsonr(original_scores, bert_scores)

print("Pearson Correlation Results for bert-base:")
print(f"Correlation coefficient (r) = {corr:.4f}")
print(f"p-value = {p_value:.4f}")

corr, p_value = pearsonr(original_scores, t5_scores)

print("Pearson Correlation Results for t5-base:")
print(f"Correlation coefficient (r) = {corr:.4f}")
print(f"p-value = {p_value:.4f}")

corr, p_value = pearsonr(original_scores, mistral_scores)

print("Pearson Correlation Results for mistral:")
print(f"Correlation coefficient (r) = {corr:.4f}")
print(f"p-value = {p_value:.4f}")