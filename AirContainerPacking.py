import collections
import os
import random
import tempfile
import uuid
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class Box:
    def __init__(self, id, length, width, height, quantity):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.quantity = quantity
        self.volume = length * width * height
        # 托盘尺寸固定为1.2x0.8
        self.pallet_length = 1.2
        self.pallet_width = 0.8

    def __repr__(self):
        return f"Box{self.id}({self.length}x{self.width}x{self.height}, qty:{self.quantity})"


class Shelf:
    def __init__(self, group_id, length, width, height, levels=4):
        self.group_id = group_id
        self.length = length
        self.width = width
        self.height = height
        self.levels = levels
        self.volume = length * width * height * levels

    def __repr__(self):
        return f"ShelfGroup{self.group_id}({self.length}x{self.width}x{self.height}, levels:{self.levels})"


class Placement:
    def __init__(self, box_id, shelf_group, level):
        self.box_id = box_id
        self.shelf_group = shelf_group
        self.level = level

    def __repr__(self):
        return f"Box{self.box_id}->Group{self.shelf_group}-Level{self.level}-{'Lengthwise'}"


class AirContainerPackingGA:
    def __init__(self, excel_file, pop_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2, elite_size=5,
                 safety_distance=0.03):  # 改为3公分安全距离
        self.df = pd.read_excel(excel_file)
        self.boxes = self._parse_box_data()
        self.safety_distance = safety_distance  # 安全距离

        # 定义货架 - 每层高度固定为1.55m，共4层
        self.shelves = [
            Shelf(1, 7.0, 1.3, 1.55, levels=4),  # 第一组货架
            Shelf(2, 8.2, 1.3, 1.55, levels=4)  # 第二组货架
        ]

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        # 创建箱子ID到对象的映射
        self.box_dict = {box.id: box for box in self.boxes}

        # 计算总库存体积
        self.total_inventory_volume = sum(box.volume * box.quantity for box in self.boxes)

        # 计算货架总体积
        self.total_shelf_volume = sum(shelf.volume for shelf in self.shelves)

    def _parse_box_data(self):
        """从Excel数据解析箱子信息（自动过滤超规格箱子）"""
        boxes = []

        MAX_HEIGHT = 1.55
        MAX_WIDTH = 1.3

        for idx, row in self.df.iterrows():
            try:
                # 解析尺寸字符串 (格式: "长*宽*高")
                dimensions_str = str(row['尺寸（M）'])
                if '*' not in dimensions_str:
                    continue

                dimensions = dimensions_str.replace(' ', '').split('*')
                if len(dimensions) != 3:
                    continue

                length = float(dimensions[0])
                width = float(dimensions[1])
                height = float(dimensions[2])

                # ========= 关键过滤条件 =========
                if height > MAX_HEIGHT or width > MAX_WIDTH:
                    continue
                # =================================

                quantity = int(row['Total Stock'])
                if quantity <= 0:
                    continue

                material = str(row['Material'])

                boxes.append(Box(material, length, width, height, quantity))

            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse row {idx}: {e}")
                continue

        return boxes

    def get_box_dimensions(self, box_id: int) -> Tuple[float, float]:
        """根据朝向获取箱子的有效长度和宽度（包含安全距离）"""
        box = self.box_dict[box_id]

        # 判断使用托盘还是箱子作为边界
        # 如果托盘比箱子大，使用托盘尺寸作为边界；否则使用箱子尺寸
        if box.pallet_length >= box.length:
            boundary_length = box.pallet_length
        else:
            boundary_length = box.length

        if box.pallet_width >= box.width:
            boundary_width = box.pallet_width
        else:
            boundary_width = box.width

        return boundary_length + self.safety_distance, boundary_width + self.safety_distance

    def create_chromosome(self) -> List[Placement]:
        """生成真正无法再放下任何箱子的最优解，并输出详细分析"""
        chromosome = []

        # 初始化使用记录
        shelf_usage = {(shelf_idx, level): 0.0 for shelf_idx, shelf in enumerate(self.shelves) for level in
                       range(shelf.levels)}
        used_counts = {box.id: 0 for box in self.boxes}
        eff_lengths = {box.id: self.get_box_dimensions(box.id)[0] for box in self.boxes}

        # 生成所有箱子实例
        all_boxes = []
        for box in self.boxes:
            for unit_index in range(box.quantity):
                all_boxes.append((box.id, unit_index, eff_lengths[box.id], box.volume))

        # 多种排序策略
        strategies = [
            lambda x: x[2],  # 长度降序
            lambda x: x[3],  # 体积降序
            lambda x: x[3] / x[2],  # 密度降序
        ]

        sort_key = random.choice(strategies)
        all_boxes.sort(key=sort_key, reverse=True)

        # 使用列表而不是堆，但维护排序
        shelves_sorted = []
        for (shelf_idx, level), used in shelf_usage.items():
            shelves_sorted.append((used, shelf_idx, level, self.shelves[shelf_idx].length))
        shelves_sorted.sort(key=lambda x: x[0])  # 按已用长度排序

        # 迭代填充直到无法放置
        changed = True
        while changed:
            changed = False

            # 尝试放置每个箱子
            for i in range(len(all_boxes)):
                if i >= len(all_boxes):
                    break

                box_id, unit_index, eff_len, volume = all_boxes[i]

                if used_counts[box_id] >= self.box_dict[box_id].quantity:
                    all_boxes.pop(i)
                    i -= 1
                    continue

                # 查找最佳货架层
                best_shelf_idx = -1
                best_level = -1
                best_gap = float('inf')

                for used, shelf_idx, level, total_len in shelves_sorted:
                    remaining = total_len - used
                    if eff_len <= remaining:
                        gap = remaining - eff_len
                        if gap < best_gap:
                            best_gap = gap
                            best_shelf_idx = shelf_idx
                            best_level = level

                if best_shelf_idx != -1:
                    # 放置箱子
                    used_counts[box_id] += 1
                    shelf_usage[(best_shelf_idx, best_level)] += eff_len

                    unique_id = f"{box_id}_{unit_index}_{uuid.uuid4().hex[:8]}"
                    chromosome.append(Placement(unique_id, best_shelf_idx, best_level))

                    all_boxes.pop(i)
                    i -= 1
                    changed = True

                    # 更新货架排序
                    shelves_sorted = []
                    for (shelf_idx, level), used in shelf_usage.items():
                        shelves_sorted.append((used, shelf_idx, level, self.shelves[shelf_idx].length))
                    shelves_sorted.sort(key=lambda x: x[0])
                    break  # 重新开始遍历

        # ==================== 输出详细分析 ====================
        print("\n" + "=" * 60)
        print("染色体生成结果分析")
        print("=" * 60)

        # 1. 输出每个货架层的使用情况
        print("\n货架层使用情况:")
        total_remaining = 0
        for shelf_idx, shelf in enumerate(self.shelves):
            for level in range(shelf.levels):
                used_length = shelf_usage.get((shelf_idx, level), 0)
                remaining = shelf.length - used_length
                total_remaining += remaining
                print(
                    f"  货架{shelf_idx}-层{level}: 已用{used_length:.2f}m, 剩余{remaining:.2f}m, 利用率{used_length / shelf.length * 100:.1f}%")

        # 2. 输出剩余箱子信息
        print(f"\n剩余箱子数量: {len(all_boxes)}")
        if all_boxes:
            min_length = min(box[2] for box in all_boxes)
            max_length = max(box[2] for box in all_boxes)
            avg_length = sum(box[2] for box in all_boxes) / len(all_boxes)
            print(f"剩余箱子最小长度: {min_length:.3f}m")
            print(f"剩余箱子最大长度: {max_length:.3f}m")
            print(f"剩余箱子平均长度: {avg_length:.3f}m")

            # 检查是否有箱子能放入剩余空间
            can_place_any = False
            for shelf_idx, shelf in enumerate(self.shelves):
                for level in range(shelf.levels):
                    remaining = shelf.length - shelf_usage.get((shelf_idx, level), 0)
                    if remaining > 0:
                        for box_id, unit_index, eff_len, volume in all_boxes:
                            if eff_len <= remaining:
                                can_place_any = True
                                break
                        if can_place_any:
                            break
                if can_place_any:
                    break

            if can_place_any:
                print("❌ 警告: 存在可以放入剩余空间的箱子，解不是最优!")
            else:
                print("✅ 验证: 所有剩余箱子都无法放入任何剩余空间")
        else:
            print("✅ 所有箱子都已放置")

        # 3. 输出总体统计
        total_capacity = sum(shelf.length * shelf.levels for shelf in self.shelves)
        total_used = sum(shelf_usage.values())
        utilization = total_used / total_capacity * 100 if total_capacity > 0 else 0

        print(f"\n总体统计:")
        print(f"总容量: {total_capacity:.2f}m")
        print(f"已使用: {total_used:.2f}m")
        print(f"剩余空间: {total_remaining:.2f}m")
        print(f"空间利用率: {utilization:.1f}%")

        # 4. 输出染色体内容（前10个放置）
        print(f"\n染色体内容 (前10个放置):")
        for i, placement in enumerate(chromosome[:10]):
            box_id = placement.box_id.split('_')[0]
            eff_len = eff_lengths[box_id]
            print(
                f"  {i + 1}. {placement.box_id} -> 货架{placement.shelf_group}-层{placement.level} (长度{eff_len:.2f}m)")

        if len(chromosome) > 10:
            print(f"  ... 还有{len(chromosome) - 10}个放置")

        print("=" * 60 + "\n")

        return chromosome
    # def create_chromosome(self) -> List[Placement]:
    #     """生成真正无法再放下任何箱子的解（位置多样性版本）"""
    #
    #     chromosome = []
    #
    #     shelf_usage = {
    #         (shelf_idx, level): 0.0
    #         for shelf_idx, shelf in enumerate(self.shelves)
    #         for level in range(shelf.levels)
    #     }
    #
    #     used_counts = {box.id: 0 for box in self.boxes}
    #     eff_lengths = {box.id: self.get_box_dimensions(box.id)[0] for box in self.boxes}
    #
    #     all_boxes = []
    #     for box in self.boxes:
    #         for unit_index in range(box.quantity):
    #             all_boxes.append((box.id, unit_index, eff_lengths[box.id], box.volume))
    #
    #     strategies = [
    #         lambda x: x[2],
    #         lambda x: x[3],
    #         lambda x: x[3] / x[2],
    #     ]
    #     sort_key = random.choice(strategies)
    #     all_boxes.sort(key=sort_key, reverse=True)
    #
    #     position_strategy = random.choice([
    #         "best_fit",
    #         "first_fit",
    #         "random_fit",
    #         "worst_fit",
    #     ])
    #
    #     def rebuild_shelves():
    #         lst = []
    #         for (shelf_idx, level), used in shelf_usage.items():
    #             lst.append((used, shelf_idx, level, self.shelves[shelf_idx].length))
    #         lst.sort(key=lambda x: x[0])
    #         return lst
    #
    #     shelves_sorted = rebuild_shelves()
    #
    #     changed = True
    #     while changed:
    #         changed = False
    #         random.shuffle(all_boxes)
    #
    #         i = 0
    #         while i < len(all_boxes):
    #             box_id, unit_index, eff_len, volume = all_boxes[i]
    #
    #             if used_counts[box_id] >= self.box_dict[box_id].quantity:
    #                 all_boxes.pop(i)
    #                 continue
    #
    #             # 小概率跳过，防止路径锁死
    #             if random.random() < 0.1:
    #                 i += 1
    #                 continue
    #
    #             # ===== 所有可放位置 =====
    #             candidates = []
    #             for used, shelf_idx, level, total_len in shelves_sorted:
    #                 remaining = total_len - used
    #                 if eff_len <= remaining:
    #                     candidates.append((remaining, shelf_idx, level))
    #
    #             if not candidates:
    #                 i += 1
    #                 continue
    #
    #             # ===== 位置多样性选择 =====
    #             if position_strategy == "best_fit":
    #                 _, shelf_idx, level = min(candidates, key=lambda x: x[0])
    #             elif position_strategy == "worst_fit":
    #                 _, shelf_idx, level = max(candidates, key=lambda x: x[0])
    #             elif position_strategy == "first_fit":
    #                 _, shelf_idx, level = candidates[0]
    #             else:
    #                 _, shelf_idx, level = random.choice(candidates)
    #
    #             # ===== 放置 =====
    #             used_counts[box_id] += 1
    #             shelf_usage[(shelf_idx, level)] += eff_len
    #
    #             unique_id = f"{box_id}_{unit_index}_{uuid.uuid4().hex[:8]}"
    #             chromosome.append(Placement(unique_id, shelf_idx, level))
    #
    #             all_boxes.pop(i)
    #             shelves_sorted = rebuild_shelves()
    #             changed = True
    #             break
    #
    #     return chromosome

    def evaluate_fitness(self, chromosome: List[Placement]) -> float:
        """评估染色体适应度：计算体积利用率（假设解都是合法的）"""
        total_used_volume = 0

        # 计算总使用体积
        for placement in chromosome:
            original_box_id = placement.box_id.split('_')[0]
            box = self.box_dict[original_box_id]
            total_used_volume += box.volume

        # 计算总可用体积
        total_available_volume = sum(
            shelf.length * shelf.width * shelf.height * shelf.levels
            for shelf in self.shelves
        )

        return total_used_volume / total_available_volume

    def selection(self, population: List[List[Placement]], fitnesses: List[float]) -> List[List[Placement]]:
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            # 随机选择3个个体进行竞争
            candidates = random.sample(list(zip(population, fitnesses)), 5)
            # 选择适应度最高的
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        """
        安全层交换交叉：
        - 仅在相同 shelf_group 内交换某一层
        - 交换后必须仍是合法解，否则放弃
        """

        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        max_trials = 30  # 防止死循环

        for _ in range(max_trials):

            # === 1. 随机选择一个货架 ===
            shelf_idx = random.choice(range(len(self.shelves)))

            shelf = self.shelves[shelf_idx]
            level = random.randint(0, shelf.levels - 1)

            # === 2. 拆分 parent 中该 shelf 的该层 ===
            def split(chromosome):
                level_items = []
                others = []
                for p in chromosome:
                    if p.shelf_group == shelf_idx and p.level == level:
                        level_items.append(p)
                    else:
                        others.append(p)
                return level_items, others

            p1_level, p1_rest = split(parent1)
            p2_level, p2_rest = split(parent2)

            # === 3. 交换该层 ===
            child1 = p1_rest + p2_level
            child2 = p2_rest + p1_level

            # === 4. 校验合法性 ===
            if self.is_legal(child1) and self.is_legal(child2):
                return child1, child2

        # 多次尝试失败，放弃交叉
        return parent1.copy(), parent2.copy()

    def is_legal(self, chromosome):
        # ---------- 1. 库存检查 ----------
        used_boxes = {}
        for p in chromosome:
            box_id = p.box_id.split("_")[0]
            used_boxes[box_id] = used_boxes.get(box_id, 0) + 1
            if used_boxes[box_id] > self.box_dict[box_id].quantity:
                return False

        # ---------- 2. 每层长度检查 ----------
        shelf_level_used = {}

        for p in chromosome:
            shelf_idx = p.shelf_group
            level = p.level
            key = (shelf_idx, level)

            if key not in shelf_level_used:
                shelf_level_used[key] = 0.0

            eff_length, _ = self.get_box_dimensions(p.box_id.split("_")[0])
            shelf_level_used[key] += eff_length

            if shelf_level_used[key] > self.shelves[shelf_idx].length + 1e-6:
                return False

        return True

    def mutation(self, chromosome: List[Placement]) -> List[Placement]:
        """严格保证：变异前后染色体始终合法"""

        if random.random() > self.mutation_rate or not chromosome:
            return chromosome.copy()

        original = chromosome
        mutated = [Placement(p.box_id, p.shelf_group, p.level) for p in chromosome]

        mutation_type = random.choices(
            population=[0, 1, 2],
            weights=[0.3, 0.4, 0.3],
            k=1
        )[0]

        # ---------- 1. 交换两个 placement ----------
        if mutation_type == 0 and len(mutated) > 1:
            for _ in range(10):
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]

                if self.is_legal(mutated):
                    return mutated

                # 回滚
                mutated[i], mutated[j] = mutated[j], mutated[i]

        # ---------- 2. 改变一个箱子的层 ----------
        elif mutation_type == 1:
            idx = random.randrange(len(mutated))
            p = mutated[idx]

            for _ in range(10):
                new_shelf = random.randrange(len(self.shelves))
                new_level = random.randrange(self.shelves[new_shelf].levels)

                old_shelf, old_level = p.shelf_group, p.level
                p.shelf_group, p.level = new_shelf, new_level

                if self.is_legal(mutated):
                    return mutated

                # 回滚
                p.shelf_group, p.level = old_shelf, old_level

        # ---------- 3. 移除并重新插入 ----------
        elif mutation_type == 2:
            idx = random.randrange(len(mutated))
            removed = mutated.pop(idx)

            for _ in range(10):
                shelf_idx = random.randrange(len(self.shelves))
                level = random.randrange(self.shelves[shelf_idx].levels)

                mutated.append(Placement(removed.box_id, shelf_idx, level))

                if self.is_legal(mutated):
                    return mutated

                mutated.pop()

            # 插不回去，彻底回滚
            return original.copy()

        # 所有尝试失败，返回原解
        return original.copy()

    def run(self):
        """强精英 + 交叉择优 + 变异择优 + 不允许退化的 GA 主循环"""

        # =======================
        # 初始化种群
        # =======================
        population = [self.create_chromosome() for _ in range(self.pop_size)]

        best_fitness = -float('inf')
        best_chromosome = None
        fitness_history = []
        stagnation_count = 0

        # =======================
        # 主循环
        # =======================
        for generation in range(self.generations):

            # -------- 评估当前种群 --------
            fitnesses = [self.evaluate_fitness(ind) for ind in population]

            current_best_idx = int(np.argmax(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            current_best = population[current_best_idx]

            # -------- 强精英更新 --------
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = [
                    Placement(p.box_id, p.shelf_group, p.level)
                    for p in current_best
                ]
                stagnation_count = 0
            else:
                stagnation_count += 1

            fitness_history.append(best_fitness)

            # =======================
            # 选择
            # =======================
            selected = self.selection(population, fitnesses)

            # =======================
            # 精英保留（深拷贝）
            # =======================
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elite = [
                [Placement(p.box_id, p.shelf_group, p.level) for p in population[i]]
                for i in elite_indices
            ]

            # =======================
            # 交叉（父代 vs 子代，择优）
            # =======================
            children = []

            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    children.append(selected[i])
                    continue

                p1 = selected[i]
                p2 = selected[i + 1]

                c1, c2 = self.crossover(p1, p2)

                # 父子择优
                if self.evaluate_fitness(c1) >= self.evaluate_fitness(p1):
                    children.append(c1)
                else:
                    children.append(p1)

                if self.evaluate_fitness(c2) >= self.evaluate_fitness(p2):
                    children.append(c2)
                else:
                    children.append(p2)

            # =======================
            # 变异（前后择优）
            # =======================
            mutated_children = []

            for child in children:
                mutated = self.mutation(child)

                if self.evaluate_fitness(mutated) >= self.evaluate_fitness(child):
                    mutated_children.append(mutated)
                else:
                    mutated_children.append(child)

            # =======================
            # 构建候选新种群
            # =======================
            candidate_population = elite + mutated_children

            if len(candidate_population) > self.pop_size:
                # 对非精英个体按适应度排序，择优保留
                scored_children = [
                    (self.evaluate_fitness(ind), ind)
                    for ind in mutated_children
                ]
                scored_children.sort(key=lambda x: x[0], reverse=True)

                candidate_population = elite + [
                    ind for _, ind in scored_children[:self.pop_size - len(elite)]
                ]
            elif len(candidate_population) < self.pop_size:
                candidate_population.extend(
                    [self.create_chromosome()
                     for _ in range(self.pop_size - len(candidate_population))]
                )

            # =======================
            # ★ 整代退化检测（最终保险）
            # =======================

            population = candidate_population

            # =======================
            # 日志
            # =======================
            if generation % 200 == 0:
                print(
                    f"Gen {generation}, "
                    f"Best: {best_fitness:.4f}, "
                    f"Avg: {np.mean(fitnesses):.4f}, "
                    f"Stagnation: {stagnation_count}"
                )

        return best_chromosome, best_fitness, fitness_history

    def decode_solution(self, chromosome: List[Placement]):
        """
        正确原则：
        1. decode 只负责“忠实还原染色体”
        2. 不再做长度 / 宽度 / 高度可行性判断（GA 已保证）
        3. 不改变同一层内的箱子顺序（保持 chromosome 顺序）
        """

        shelf_usage = {}
        total_available_volume = 0.0
        total_used_volume = 0.0

        # ================= 初始化 =================
        for shelf_idx, shelf in enumerate(self.shelves):
            for level in range(shelf.levels):
                shelf_usage[(shelf_idx, level)] = {
                    "used_length": 0.0,
                    "used_volume": 0.0,
                    "boxes": []
                }
                total_available_volume += shelf.length * shelf.width * shelf.height

        used_boxes = {}

        # ================= 还原染色体 =================
        for placement in chromosome:
            original_box_id = placement.box_id.split("_")[0]
            box = self.box_dict[original_box_id]

            if used_boxes.get(original_box_id, 0) >= box.quantity:
                continue

            shelf_idx = placement.shelf_group
            level = placement.level
            level_info = shelf_usage[(shelf_idx, level)]

            effective_length, effective_width = self.get_box_dimensions(original_box_id)

            level_info["used_length"] += effective_length
            level_info["used_volume"] += box.volume
            total_used_volume += box.volume

            level_info["boxes"].append({
                "box": box,
                "unique_id": placement.box_id,
                "effective_length": effective_length,
                "effective_width": effective_width,
                "actual_length": box.length,
                "actual_width": box.width,
                "height": box.height,
                "safety_distance": self.safety_distance,
                "pallet_length": box.pallet_length,
                "pallet_width": box.pallet_width
            })

            used_boxes[original_box_id] = used_boxes.get(original_box_id, 0) + 1

        volume_utilization = (
            total_used_volume / total_available_volume
            if total_available_volume > 0 else 0
        )

        # ==================== 统计输出 ====================
        print("\n" + "=" * 60)
        print("解码结果统计分析")
        print("=" * 60)

        # -------- 每层长度统计 --------
        total_capacity_length = 0.0
        total_used_length = 0.0
        total_remaining_length = 0.0

        print("\n各货架层长度使用情况:")
        for shelf_idx, shelf in enumerate(self.shelves):
            for level in range(shelf.levels):
                used_len = shelf_usage[(shelf_idx, level)]["used_length"]
                remaining = shelf.length - used_len

                total_capacity_length += shelf.length
                total_used_length += used_len
                total_remaining_length += remaining

                utilization = used_len / shelf.length * 100 if shelf.length > 0 else 0

                print(
                    f"  货架{shelf_idx}-层{level}: "
                    f"已用 {used_len:.3f} m, "
                    f"剩余 {remaining:.3f} m, "
                    f"利用率 {utilization:.1f}%"
                )

        # -------- 剩余箱子统计 --------
        remaining_boxes = []
        for box in self.boxes:
            remaining_qty = box.quantity - used_boxes.get(box.id, 0)
            if remaining_qty > 0:
                eff_len, _ = self.get_box_dimensions(box.id)
                for _ in range(remaining_qty):
                    remaining_boxes.append(eff_len)

        print("\n剩余箱子统计:")
        if remaining_boxes:
            print(f"  剩余箱子数量: {len(remaining_boxes)}")
            print(f"  剩余箱子最小长度: {min(remaining_boxes):.3f} m")
            print(f"  剩余箱子最大长度: {max(remaining_boxes):.3f} m")
            print(
                f"  剩余箱子平均长度: "
                f"{sum(remaining_boxes) / len(remaining_boxes):.3f} m"
            )

            # 是否理论可放（仅提示）
            can_fit = False
            for shelf_idx, shelf in enumerate(self.shelves):
                for level in range(shelf.levels):
                    remaining = shelf.length - shelf_usage[(shelf_idx, level)]["used_length"]
                    if any(box_len <= remaining for box_len in remaining_boxes):
                        can_fit = True
                        break
                if can_fit:
                    break

            if can_fit:
                print("  ⚠️ 注意：存在理论可放的剩余箱子（GA 决策结果）")
            else:
                print("  ✅ 验证：所有剩余箱子均无法放入任何剩余空间")
        else:
            print("  ✅ 所有箱子均已放置")

        # -------- 总体统计 --------
        print("\n总体统计:")
        print(f"  总可用长度: {total_capacity_length:.3f} m")
        print(f"  已使用长度: {total_used_length:.3f} m")
        print(f"  剩余长度: {total_remaining_length:.3f} m")
        print(f"  长度利用率: {total_used_length / total_capacity_length * 100:.1f}%")
        print(f"  体积利用率: {volume_utilization * 100:.1f}%")

        print("=" * 60 + "\n")

        return shelf_usage, used_boxes, total_used_volume, volume_utilization


def create_box_mesh(x_pos, y_pos, z_pos, length, width, height, color):
    """创建完整箱子的3D网格 - 确保长边与X轴平行"""
    # 定义箱子的8个顶点
    vertices = np.array([
        [x_pos, y_pos, z_pos],  # 0: 左下前
        [x_pos + length, y_pos, z_pos],  # 1: 右下前
        [x_pos + length, y_pos + width, z_pos],  # 2: 右后前
        [x_pos, y_pos + width, z_pos],  # 3: 左后前
        [x_pos, y_pos, z_pos + height],  # 4: 左下后
        [x_pos + length, y_pos, z_pos + height],  # 5: 右下后
        [x_pos + length, y_pos + width, z_pos + height],  # 6: 右后后
        [x_pos, y_pos + width, z_pos + height]  # 7: 左后后
    ])

    # 定义箱子的6个面（每个面由2个三角形组成）
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 5, 6], [4, 6, 7],  # 顶面
        [0, 1, 5], [0, 5, 4],  # 前面
        [2, 3, 7], [2, 7, 6],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 2, 6], [1, 6, 5]  # 右面
    ])

    return vertices, faces, color


def visualize_3d_shelf_layout(shelf_usage, shelves):
    """为每个货架组创建3D可视化 - 确保长边与X轴平行"""
    figures = []

    # 使用鲜明的颜色方案
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                     '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']

    for shelf_idx in range(len(shelves)):
        fig = go.Figure()
        shelf = shelves[shelf_idx]
        shelf_height = shelf.height
        shelf_width = shelf.width
        shelf_length = shelf.length

        # 绘制货架立柱
        post_positions = [
            (0, 0), (0, shelf_width), (shelf_length, 0), (shelf_length, shelf_width)
        ]

        for x, y in post_positions:
            for level in range(shelf.levels + 1):
                z_bottom = level * shelf_height
                z_top = z_bottom + 0.1  # 立柱高度

                # 立柱
                fig.add_trace(go.Mesh3d(
                    x=[x, x + 0.1, x + 0.1, x] * 2,
                    y=[y, y, y + 0.1, y + 0.1] * 2,
                    z=[z_bottom, z_bottom, z_bottom, z_bottom,
                       z_top, z_top, z_top, z_top],
                    i=[0, 0, 0, 0, 5, 5],
                    j=[1, 2, 3, 4, 6, 7],
                    k=[2, 3, 4, 1, 7, 4],
                    color='#8B4513',  # 棕色
                    opacity=0.9,
                    flatshading=True,
                    showlegend=False
                ))

        # 绘制货架层板
        for level in range(shelf.levels):
            z_pos = level * shelf_height

            # 层板
            fig.add_trace(go.Mesh3d(
                x=[0, shelf_length, shelf_length, 0],
                y=[0, 0, shelf_width, shelf_width],
                z=[z_pos, z_pos, z_pos, z_pos],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color='#D2B48C',  # 浅木色
                opacity=0.7,
                flatshading=True,
                showlegend=False
            ))

            # 层板边缘
            fig.add_trace(go.Scatter3d(
                x=[0, shelf_length, shelf_length, 0, 0],
                y=[0, 0, shelf_width, shelf_width, 0],
                z=[z_pos, z_pos, z_pos, z_pos, z_pos],
                mode='lines',
                line=dict(color='#8B4513', width=3),
                showlegend=False
            ))

        # 为每个层级添加箱子和托盘
        for level in range(shelf.levels):
            if (shelf_idx, level) in shelf_usage:
                info = shelf_usage[(shelf_idx, level)]
                x_pos = 0.1  # 从货架边缘开始放置
                z_pos = level * shelf_height + 0.01  # 放在层板上方

                for i, box_info in enumerate(info['boxes']):
                    box_length = box_info['box'].length
                    box_width = box_info['box'].width

                    box_height = box_info['height']
                    pallet_length = box_info['pallet_length']
                    pallet_width = box_info['pallet_width']

                    # 获取有效尺寸（包含安全距离）- 用于布局计算
                    effective_length = box_info['effective_length']
                    effective_width = box_info['effective_width']

                    # 确定箱子在托盘上的位置（居中放置）
                    # 箱子应该居中放置在托盘上，使用箱子的实际尺寸
                    box_x_offset = (pallet_length - box_length) / 2
                    box_y_offset = (pallet_width - box_width) / 2

                    # 确保箱子不会超出托盘边界
                    box_x_offset = max(0, box_x_offset)  # 如果箱子比托盘大，从边缘开始
                    box_y_offset = max(0, box_y_offset)

                    # 选择颜色
                    color_idx = hash(box_info['box'].id) % len(color_palette)
                    box_color = color_palette[color_idx]
                    pallet_color = '#A0522D'  # 托盘颜色为棕色

                    # 先绘制托盘
                    pallet_vertices, pallet_faces, _ = create_box_mesh(
                        x_pos, 0.1, z_pos - 0.05,  # 托盘在箱子下方
                        pallet_length, pallet_width, 0.05,  # 托盘高度设为5cm
                        pallet_color
                    )
                    fig.add_trace(go.Mesh3d(
                        x=pallet_vertices[:, 0],
                        y=pallet_vertices[:, 1],
                        z=pallet_vertices[:, 2],
                        i=pallet_faces[:, 0],
                        j=pallet_faces[:, 1],
                        k=pallet_faces[:, 2],
                        color=pallet_color,
                        opacity=0.8,
                        flatshading=True,
                        name='托盘' if i == 0 else None,
                        showlegend=True if i == 0 else False
                    ))

                    # 创建完整的箱子网格（使用箱子的实际尺寸）
                    vertices, faces, color = create_box_mesh(
                        x_pos + box_x_offset,  # 箱子在托盘上居中
                        0.1 + box_y_offset,  # 箱子在托盘上居中
                        z_pos,  # 放在托盘上方
                        box_length,  # 箱子的实际长度
                        box_width,  # 箱子的实际宽度
                        box_height,  # 箱子的实际高度
                        box_color
                    )

                    # 添加箱子
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=0.8,
                        flatshading=True,
                        name=box_info['box'].id if i == 0 else None,
                        showlegend=True if i == 0 else False
                    ))

                    # 添加箱子ID标签
                    fig.add_trace(go.Scatter3d(
                        x=[x_pos + pallet_length / 2],
                        y=[0.1 + pallet_width / 2],
                        z=[z_pos + box_height / 2],
                        mode='text',
                        text=[
                            f"{box_info['box'].id}<br>{box_info['box'].length}×{box_info['box'].width}×{box_info['box'].height}"],
                        textposition='middle center',
                        textfont=dict(size=12, color='black', weight='bold'),
                        showlegend=False
                    ))

                    # 使用 effective_length 递增 x_pos，确保无重叠
                    x_pos += effective_length

        # 更新3D场景设置
        fig.update_layout(
            title=dict(
                text=f"货架组 {shelf_idx + 1} 3D布局图",
                x=0.5,
                font=dict(size=20, color="darkblue")
            ),
            width=1000,
            height=800,
            scene=dict(
                xaxis=dict(title='长度 (X轴)', range=[-0.5, shelf_length + 0.5]),
                yaxis=dict(title='宽度 (Y轴)', range=[-0.5, shelf_width + 0.5]),
                zaxis=dict(title='高度 (Z轴)', range=[-0.5, shelf_height * shelf.levels + 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1, z=1.5),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=True
        )
        figures.append(fig)

    return figures


def visualize_front_view(shelf_usage, shelves):
    """创建正视图 - 显示货架和箱子的正面"""
    figures = []

    for shelf_idx in range(len(shelves)):
        fig = go.Figure()
        shelf = shelves[shelf_idx]
        shelf_height = shelf.height
        shelf_length = shelf.length

        # 绘制货架正视图
        # 绘制货架立柱（正面视角）
        for x in [0, shelf_length]:
            for level in range(shelf.levels + 1):
                z_bottom = level * shelf_height
                z_top = z_bottom + 0.1

                fig.add_trace(go.Scatter(
                    x=[x, x + 0.1, x + 0.1, x, x],
                    y=[z_bottom, z_bottom, z_top, z_top, z_bottom],
                    fill="toself",
                    fillcolor='#8B4513',
                    line=dict(color='#8B4513', width=2),
                    name='立柱' if x == 0 and level == 0 else '',
                    showlegend=False
                ))

        # 绘制货架层板（正面视角）- 最底层
        for level in range(shelf.levels):
            z_pos = level * shelf_height

            fig.add_trace(go.Scatter(
                x=[0, shelf_length, shelf_length, 0, 0],
                y=[z_pos, z_pos, z_pos + 0.05, z_pos + 0.05, z_pos],
                fill="toself",
                fillcolor='#D2B48C',
                line=dict(color='#8B4513', width=2),
                name='层板' if level == 0 else '',
                showlegend=False
            ))

        # 为每个层级添加箱子和托盘（正面视角）
        for level in range(shelf.levels):
            if (shelf_idx, level) in shelf_usage:
                info = shelf_usage[(shelf_idx, level)]
                z_pos = level * shelf_height + 0.05  # 层板顶部位置
                pallet_height = 0.1  # 托盘高度

                # 从左到右排列
                x_pos = 0.01  # 从货架左侧开始放置

                for i, box_info in enumerate(info['boxes']):
                    # 根据朝向获取正确的尺寸

                    actual_length = box_info['box'].length
                    actual_width = box_info['box'].width

                    # 考虑托盘尺寸
                    actual_length = max(actual_length, box_info['pallet_length'])
                    actual_width = max(actual_width, box_info['pallet_width'])

                    actual_height = box_info['height']
                    pallet_length = box_info['pallet_length']
                    pallet_width = box_info['pallet_width']

                    # 获取有效尺寸（包含安全距离）
                    effective_length = box_info['effective_length']

                    # 确定箱子在托盘上的位置（箱子在托盘上居中）
                    if pallet_length >= actual_length:
                        box_x_offset = (pallet_length - actual_length) / 2
                    else:
                        box_x_offset = 0

                    # 选择颜色
                    color_idx = hash(box_info['box'].id) % 10
                    box_color = px.colors.qualitative.Set3[color_idx]
                    pallet_color = '#A0522D'

                    # 绘制托盘（托盘在货物下方）
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos + pallet_length, x_pos + pallet_length, x_pos, x_pos],
                        y=[z_pos, z_pos, z_pos + pallet_height, z_pos + pallet_height, z_pos],
                        fill="toself",
                        fillcolor=pallet_color,
                        line=dict(color='black', width=2),
                        name='托盘' if i == 0 else '',
                        showlegend=False
                    ))

                    # 绘制箱子正面（箱子在托盘上方）
                    fig.add_trace(go.Scatter(
                        x=[x_pos + box_x_offset,
                           x_pos + box_x_offset + actual_length,
                           x_pos + box_x_offset + actual_length,
                           x_pos + box_x_offset,
                           x_pos + box_x_offset],
                        y=[z_pos + pallet_height,
                           z_pos + pallet_height,
                           z_pos + pallet_height + actual_height,
                           z_pos + pallet_height + actual_height,
                           z_pos + pallet_height],
                        fill="toself",
                        fillcolor=box_color,
                        line=dict(color='black', width=2),
                        name=box_info['box'].id,
                        showlegend=False
                    ))

                    # 添加货物标注：料号及尺寸（换行显示）
                    fig.add_annotation(
                        x=x_pos + pallet_length / 2,
                        y=z_pos + pallet_height + actual_height / 2,
                        text=f"{box_info['box'].id}<br>{actual_length:.2f}×{actual_width:.2f}×{actual_height:.2f}",
                        showarrow=False,
                        font=dict(size=10, color='black', weight='bold'),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        opacity=0.9
                    )

                    # 使用 effective_length 递增 x_pos（从左到右排列）
                    x_pos += effective_length

        # 添加货架尺寸标注（统一在右侧）
        fig.add_annotation(
            x=shelf_length + 0.3,
            y=shelf_height * shelf.levels,
            text=f"货架尺寸<br>{shelf_length:.2f}×{shelf.width:.2f}×{shelf_height:.2f}",
            showarrow=False,
            font=dict(size=18, color='darkgreen', weight='bold'),
            textangle=0,
            bgcolor="white",
            bordercolor="darkgreen",
            borderwidth=1,
            align="center"
        )

        # 更新布局设置
        fig.update_layout(
            title=dict(
                text=f"货架组 {shelf_idx + 1} 正视图",
                x=0.5,
                font=dict(size=20, color="darkblue")
            ),
            width=1000,
            height=600,
            xaxis=dict(
                title='长度 (m)',
                range=[-0.2, shelf_length + 0.8],
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                title='高度 (m)',
                range=[-0.5, shelf_height * shelf.levels + 0.5],
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            showlegend=True,
            plot_bgcolor='white'
        )
        figures.append(fig)

    return figures


def main():
    st.set_page_config(page_title="航空箱货架布局优化模型", layout="wide")
    st.title("📦 航空箱货架布局优化模型")

    # 移除侧边栏参数配置，使用固定参数
    pop_size = 300
    generations = 600
    crossover_rate = 0.9
    mutation_rate = 0.1
    elite_size = 1
    safety_distance = 0.03  # 3公分安全距离

    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx"], help="请上传包含箱子尺寸和库存数据的Excel文件")

    # ================= Session State =================
    if "optimized" not in st.session_state:
        st.session_state.optimized = False
    if "ga" not in st.session_state:
        st.session_state.ga = None
    if "shelf_usage" not in st.session_state:
        st.session_state.shelf_usage = None
    if "used_boxes" not in st.session_state:
        st.session_state.used_boxes = None
    if "volume_utilization" not in st.session_state:
        st.session_state.volume_utilization = None

    if uploaded_file is not None:
        try:
            # ---------- 临时文件 ----------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # ---------- 数据预览 ----------
            st.subheader("数据预览")
            df = pd.read_excel(tmp_file_path)
            st.dataframe(df.head(), use_container_width=True)

            # ---------- 执行优化 ----------
            if st.button("开始优化", type="primary") or st.session_state.optimized:
                if not st.session_state.optimized:
                    with st.spinner("正在进行遗传算法优化..."):
                        ga = AirContainerPackingGA(
                            tmp_file_path,
                            pop_size=pop_size,
                            generations=generations,
                            crossover_rate=crossover_rate,
                            mutation_rate=mutation_rate,
                            elite_size=elite_size,
                            safety_distance=safety_distance
                        )

                        best_solution, best_fitness, history = ga.run()
                        shelf_usage, used_boxes, total_used_volume, volume_utilization = (
                            ga.decode_solution(best_solution)
                        )

                        st.session_state.optimized = True
                        st.session_state.ga = ga
                        st.session_state.shelf_usage = shelf_usage
                        st.session_state.used_boxes = used_boxes
                        st.session_state.volume_utilization = volume_utilization
                else:
                    ga = st.session_state.ga
                    shelf_usage = st.session_state.shelf_usage
                    volume_utilization = st.session_state.volume_utilization

                st.success("优化完成")

                # ================= 核心指标（长度为主） =================
                total_capacity_length = 0.0
                total_used_length = 0.0

                for shelf_idx, shelf in enumerate(ga.shelves):
                    for level in range(shelf.levels):
                        total_capacity_length += shelf.length
                        total_used_length += shelf_usage[(shelf_idx, level)]["used_length"]

                length_utilization = (
                    total_used_length / total_capacity_length
                    if total_capacity_length > 0 else 0
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("长度利用率", f"{length_utilization:.1%}")
                with col2:
                    st.metric("体积利用率", f"{volume_utilization:.1%}")


                # # 显示3D可视化
                # st.subheader("3D货架布局图")
                # figures_3d = visualize_3d_shelf_layout(shelf_usage, ga.shelves)
                # for i, fig in enumerate(figures_3d):
                #     st.plotly_chart(fig, use_container_width=True)

                # 显示正视图
                st.subheader("货架正视图")
                figures_front = visualize_front_view(shelf_usage, ga.shelves)
                for i, fig in enumerate(figures_front):
                    st.plotly_chart(fig, use_container_width=True)

                # ================= 各层长度使用情况 =================
                st.subheader("各货架层长度使用情况")

                layer_rows = []
                for shelf_idx, shelf in enumerate(ga.shelves):
                    for level in range(shelf.levels):
                        used_len = shelf_usage[(shelf_idx, level)]["used_length"]
                        remaining = shelf.length - used_len
                        utilization = used_len / shelf.length if shelf.length > 0 else 0

                        layer_rows.append({
                            "货架组": shelf_idx + 1,
                            "层级": level + 1,
                            "已用长度 (m)": f"{used_len:.3f}",
                            "剩余长度 (m)": f"{remaining:.3f}",
                            "长度利用率": f"{utilization:.1%}"
                        })

                layer_df = pd.DataFrame(layer_rows)
                st.dataframe(layer_df, use_container_width=True)

                # 详细放置方案表格
                st.subheader("详细放置方案")

                placement_data = []
                for (shelf_idx, level), info in shelf_usage.items():
                    for box_info in info['boxes']:
                        placement_data.append({
                            "货架组": shelf_idx + 1,
                            "层级": level + 1,
                            "箱子ID": box_info['box'].id,
                            "尺寸": f"{box_info['actual_length']}×{box_info['actual_width']}×{box_info['height']}m",
                            "体积": f"{box_info['box'].volume:.2f} m³",
                            "安全距离": f"{ga.safety_distance}m",
                            "托盘尺寸": f"{box_info['pallet_length']}×{box_info['pallet_width']}m"
                        })

                placement_df = pd.DataFrame(placement_data)
                st.dataframe(placement_df, use_container_width=True)

                # 清理临时文件
                os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"处理文件时出错: {str(e)}")
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)


if __name__ == "__main__":
    main()
