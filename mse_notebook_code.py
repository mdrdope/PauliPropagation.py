# ֱ���� notebook ��ʹ�õ� MSE �������
import numpy as np
import matplotlib.pyplot as plt

def compute_truncation_mse_simple(prop_2d, sampled_last_paulis, weight_exceeded_details, 
                                 coeff_sqs, nx, ny):
    """
    ���㲻ͬ truncation �����µ� MSE
    
    ��ʽ: MSE^(k) = (1/M) * ��|C_i|^2 * <P_i, ��_0>^2 * 1{wt(P_i) > k}
    ���� ��_0 = |0...0?
    """
    
    M = len(sampled_last_paulis)
    n_qubits = nx * ny
    
    # ���� |0...0? ̬��ǩ
    product_label = '0' * n_qubits
    
    print(f"���� {M} �� Pauli ���� ��_0 = |{product_label}? ������ֵ...")
    
    # ����ÿ�� Pauli �������ֵ <P_i, ��_0>
    expectations = []
    for i, pauli in enumerate(sampled_last_paulis):
        if i % 1000 == 0:
            print(f"����: {i}/{M}")
        
        exp_val = prop_2d.expectation_pauli_sum([pauli], product_label)
        expectations.append(exp_val)
    
    expectations = np.array(expectations)
    coeff_sqs = np.array(coeff_sqs)
    weight_exceeded_details = np.array(weight_exceeded_details, dtype=bool)
    
    print("���㲻ͬ truncation ����� MSE...")
    
    # ����ÿ�� truncation ���� k = 1, 2, 3, 4, 5, 6 �� MSE
    mse_results = {}
    
    for k in range(1, 7):
        # weight_exceeded_details[:, k-1] ���� weight > k �Ĳ�������
        truncated_mask = weight_exceeded_details[:, k-1]
        
        # MSE^(k) = (1/M) * ��|C_i|^2 * <P_i, ��_0>^2 * 1{wt(P_i) > k}
        mse_terms = coeff_sqs * (expectations ** 2) * truncated_mask
        mse_k = np.sum(mse_terms) / M
        
        mse_results[k] = mse_k
        
        n_truncated = np.sum(truncated_mask)
        print(f"k={k}: {n_truncated}/{M} ��ض�, MSE = {mse_k:.2e}")
    
    return mse_results, expectations

def plot_mse_results(mse_results):
    """���� MSE vs truncation level"""
    
    k_values = list(mse_results.keys())
    mse_values = list(mse_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(k_values, mse_values, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Truncation Level k')
    plt.ylabel('MSE^(k)')
    plt.title('Mean Square Error vs Truncation Level')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # �ڵ��������ֵ��ǩ
    for k, mse in zip(k_values, mse_values):
        plt.annotate(f'{mse:.1e}', (k, mse), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return k_values, mse_values

# ==========================================
# ʹ�÷���������� notebook �����У�:
# ==========================================

# 1. ���� MSE
# mse_results, expectations = compute_truncation_mse_simple(
#     prop_2d, sampled_last_paulis, weight_exceeded_details, 
#     coeff_sqs, nx, ny
# )

# 2. ���ƽ��
# plot_mse_results(mse_results)

# 3. ��ӡ������ֵ
# print("\n=== MSE ��� ===")
# for k, mse in mse_results.items():
#     print(f"MSE^({k}) = {mse:.6e}")

# 4. �����ض�ͳ��
# weight_exceeded_details_array = np.array(weight_exceeded_details, dtype=bool)
# M = len(weight_exceeded_details)
# print("\n=== �ض�ͳ�� ===")
# for k in range(1, 7):
#     n_truncated = np.sum(weight_exceeded_details_array[:, k-1])
#     percentage = n_truncated / M * 100
#     print(f"k={k}: {n_truncated:5d}/{M} ·�����ض� ({percentage:5.1f}%)") 