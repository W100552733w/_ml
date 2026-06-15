import random

# 城市之間的距離矩陣 (假設有 5 個城市, 0 到 4)
distance_matrix = [
    [0, 10, 8, 9, 7],
    [10, 0, 10, 5, 6],
    [8, 10, 0, 8, 9],
    [9, 5, 8, 0, 6],
    [7, 6, 9, 6, 0]
]

num_cities = len(distance_matrix)

# 1. 計算路徑總距離
def get_total_distance(solution):
    dist = 0
    for i in range(len(solution)):
        u = solution[i]
        v = solution[(i + 1) % len(solution)] # 回到起點
        dist += distance_matrix[u][v]
    return dist

# 2. 高度函數 (距離越短，高度越高)
def height(solution):
    return get_total_distance(solution) * -1

# 3. 鄰居函數：隨機產生一個鄰近的解 (藉由反轉路徑中的某一段)
def neighbor(solution):
    new_sol = solution.copy()
    i, j = sorted(random.sample(range(len(solution)), 2))
    new_sol[i:j+1] = reversed(new_sol[i:j+1])
    return new_sol

# 4. 爬山演算法主體
def hill_climbing():
    current_sol = list(range(num_cities))
    current_height = height(current_sol)
    
    print(f"初始解: {current_sol}, 總距離: {-current_height}")
    
    max_fail_attempts = 100
    while True:
        improved = False
        for _ in range(max_fail_attempts):
            next_sol = neighbor(current_sol)
            next_height = height(next_sol)
            
            if next_height > current_height:
                current_sol = next_sol
                current_height = next_height
                improved = True
                break
        if not improved:
            break
            
    return current_sol, -current_height

# 執行演算法
best_path, min_dist = hill_climbing()
print("\n--- 執行結果 ---")
print(f"最佳路線: {best_path}")
print(f"最短距離: {min_dist}")
