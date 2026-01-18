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
    def __init__(self, box_id, shelf_group, level, orientation):
        self.box_id = box_id
        self.shelf_group = shelf_group
        self.level = level
        self.orientation = orientation  # 0: lengthwise, 1: widthwise

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

        # 极值标准化参数（用于适应度计算）
        self.max_fitness = 0
        self.min_fitness = 0

        # 计算总库存体积
        self.total_inventory_volume = sum(box.volume * box.quantity for box in self.boxes)

        # 计算货架总体积
        self.total_shelf_volume = sum(shelf.volume for shelf in self.shelves)

        print(f"Loaded {len(self.boxes)} box types from Excel file")
        print(f"Total inventory volume: {self.total_inventory_volume:.2f}")
        print(f"Total shelf volume: {self.total_shelf_volume:.2f}")
        print(f"Max possible utilization: {min(1.0, self.total_inventory_volume / self.total_shelf_volume):.2%}")

    def _parse_box_data(self):
        """从Excel数据解析箱子信息"""
        boxes = []

        for idx, row in self.df.iterrows():
            try:
                # 解析尺寸字符串 (格式: "长*宽*高")
                dimensions_str = str(row['尺寸（M）'])
                if '*' in dimensions_str:
                    # 处理可能的空格和特殊字符
                    dimensions = dimensions_str.replace(' ', '').split('*')
                    if len(dimensions) == 3:
                        length = float(dimensions[0])
                        width = float(dimensions[1])
                        height = float(dimensions[2])

                        # 获取数量
                        quantity = int(row['Total Stock'])

                        # 使用Material作为唯一标识符
                        material = str(row['Material'])

                        boxes.append(Box(material, length, width, height, quantity))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse row {idx}: {e}")
                continue

        return boxes

    @staticmethod
    def determine_orientation(box: Box) -> int:
        """根据规则确定箱子朝向 - 所有货物都需要长边朝外"""
        # 所有货物都长边朝外
        return 0  # 长边朝外

    def get_box_dimensions(self, box_id: int, orientation: int) -> Tuple[float, float]:
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

        if orientation == 0:  # 长边朝外
            return boundary_length + self.safety_distance, boundary_width + self.safety_distance
        else:  # 宽边朝外
            return boundary_width + self.safety_distance, boundary_length + self.safety_distance

    def get_actual_box_dimensions(self, box_id: int, orientation: int) -> Tuple[float, float]:
        """获取箱子的实际尺寸（包含托盘边界但不包含安全距离）"""
        box = self.box_dict[box_id]

        # 判断使用托盘还是箱子作为边界
        if box.pallet_length >= box.length:
            boundary_length = box.pallet_length
        else:
            boundary_length = box.length

        if box.pallet_width >= box.width:
            boundary_width = box.pallet_width
        else:
            boundary_width = box.width

        if orientation == 0:  # 长边朝外
            return boundary_length, boundary_width
        else:  # 宽边朝外
            return boundary_width, boundary_length

    def create_chromosome(self) -> List[Placement]:
        """创建随机染色体"""
        chromosome = []

        # 为每个箱子类型尝试分配位置
        for box in self.boxes:
            # 对于每个箱子的每个库存单位
            for unit_index in range(box.quantity):
                # 随机选择货架组和层
                shelf_group = random.choice(range(len(self.shelves)))
                level = random.choice(range(self.shelves[shelf_group].levels))

                # 根据规则确定朝向 - 所有货物都长边朝外
                orientation = self.determine_orientation(box)

                # 创建唯一标识符（箱子ID + 单位索引 + UUID）
                unique_id = f"{box.id}_{unit_index}_{uuid.uuid4().hex[:8]}"
                chromosome.append(Placement(unique_id, shelf_group, level, orientation))

        return chromosome

    def evaluate_fitness(self, chromosome: List[Placement]) -> float:
        """评估染色体适应度（基于总体积利用率）"""
        # 初始化货架状态
        shelf_usage = {}
        total_available_volume = 0
        total_used_volume = 0

        for shelf_idx, shelf in enumerate(self.shelves):
            for level in range(shelf.levels):
                shelf_usage[(shelf_idx, level)] = {
                    'used_length': 0,
                    'used_volume': 0,
                    'boxes': []
                }
                total_available_volume += shelf.length * shelf.width * shelf.height

        # 统计使用的箱子
        used_boxes = {}
        constraint_violations = 0

        # 处理每个放置决策
        for placement in chromosome:
            # 从唯一ID中提取原始箱子ID
            original_box_id = placement.box_id.split('_')[0]
            box = self.box_dict[original_box_id]

            shelf = self.shelves[placement.shelf_group]
            level_info = shelf_usage[(placement.shelf_group, placement.level)]

            # 获取有效尺寸（包含安全距离和托盘边界）
            effective_length, effective_width = self.get_box_dimensions(original_box_id, placement.orientation)

            # 检查约束
            # 1. 宽度约束（包含安全距离和托盘/箱子边界）
            if effective_width > shelf.width:
                constraint_violations += 10  # 严重违反
                continue

            # 2. 长度约束（包含安全距离和托盘/箱子边界）
            if level_info['used_length'] + effective_length > shelf.length:
                constraint_violations += 5  # 中等违反
                continue

            # 极值标准化
            if self.max_fitness < level_info['used_length'] + effective_length:
                self.max_fitness = level_info['used_length'] + effective_length
            if self.min_fitness > level_info['used_length'] + effective_length:
                self.min_fitness = level_info['used_length'] + effective_length

            # 3. 高度约束（每层高度固定为1.55m）
            if box.height > self.shelves[0].height:  # 所有货架层高相同
                constraint_violations += 10  # 严重违反
                continue

            # 4. 库存约束（检查是否超量使用）
            box_count = used_boxes.get(original_box_id, 0)
            if box_count >= box.quantity:
                constraint_violations += 8  # 严重违反
                continue

            # 如果所有约束满足，记录放置
            level_info['used_length'] += effective_length
            level_info['used_volume'] += box.volume
            level_info['boxes'].append(placement)
            used_boxes[original_box_id] = used_boxes.get(original_box_id, 0) + 1
            total_used_volume += box.volume

        # 计算总体积
        volume_utilization = total_used_volume / total_available_volume if total_available_volume > 0 else 0

        # 计算适应度（体积利用率 - 约束违反惩罚）
        fitness = volume_utilization - (constraint_violations * 0.01)

        return max(0, fitness)  # 确保适应度非负

    def selection(self, population: List[List[Placement]], fitnesses: List[float]) -> List[List[Placement]]:
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            # 随机选择3个个体进行竞争
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # 选择适应度最高的
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1: List[Placement], parent2: List[Placement]) -> Tuple[List[Placement], List[Placement]]:
        """单点交叉"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # 选择交叉点
        min_length = min(len(parent1), len(parent2))
        if min_length <= 1:
            return parent1, parent2

        crossover_point = random.randint(1, min_length - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutation(self, chromosome: List[Placement]) -> List[Placement]:
        """变异操作"""
        if random.random() > self.mutation_rate or len(chromosome) == 0:
            return chromosome

        mutated = chromosome.copy()

        # 随机选择变异类型
        mutation_type = random.choice([0, 1])

        if mutation_type == 0 and len(mutated) > 1:  # 交换两个基因
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        elif mutation_type == 1:  # 改变放置位置
            idx = random.randint(0, len(mutated) - 1)
            placement = mutated[idx]
            placement.shelf_group = random.choice(range(len(self.shelves)))
            placement.level = random.choice(range(self.shelves[placement.shelf_group].levels))

        # 移除朝向变异，因为所有箱子必须长边朝外

        return mutated

    def run(self):
        """运行遗传算法"""
        # 初始化种群
        population = [self.create_chromosome() for _ in range(self.pop_size)]
        best_fitness = -float('inf')
        best_chromosome = None
        fitness_history = []

        for generation in range(self.generations):
            # 评估适应度
            fitnesses = [self.evaluate_fitness(ind) for ind in population]

            # 记录最佳个体
            current_best_fitness = max(fitnesses)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_idx = fitnesses.index(current_best_fitness)
                best_chromosome = population[best_idx].copy()

            fitness_history.append(current_best_fitness)

            # 选择
            selected = self.selection(population, fitnesses)

            # 精英保留
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]

            # 交叉
            children = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i + 1])
                    children.extend([child1, child2])
                else:
                    children.append(selected[i])

            # 变异
            mutated_children = [self.mutation(child) for child in children]

            # 形成新一代种群
            population = elite + mutated_children

            if generation % 50 == 0:
                print(f"Generation {generation}, Best Fitness: {current_best_fitness:.4f}")

        return best_chromosome, best_fitness, fitness_history

    def decode_solution(self, chromosome: List[Placement]):
        """解码最佳染色体，生成详细的放置方案"""
        shelf_usage = {}
        total_available_volume = 0
        total_used_volume = 0

        for shelf_idx, shelf in enumerate(self.shelves):
            for level in range(shelf.levels):
                shelf_usage[(shelf_idx, level)] = {
                    'used_length': 0,
                    'used_volume': 0,
                    'boxes': []
                }
                total_available_volume += shelf.length * shelf.width * shelf.height

        used_boxes = {}

        for placement in chromosome:
            # 从唯一ID中提取原始箱子ID
            original_box_id = placement.box_id.split('_')[0]
            box = self.box_dict[original_box_id]
            shelf = self.shelves[placement.shelf_group]
            level_info = shelf_usage[(placement.shelf_group, placement.level)]

            effective_length, effective_width = self.get_box_dimensions(original_box_id, placement.orientation)

            # 检查约束
            if (effective_width > shelf.width or
                    level_info['used_length'] + effective_length > shelf.length or
                    box.height > shelf.height or  # 高度约束
                    used_boxes.get(original_box_id, 0) >= box.quantity):
                continue  # 跳过违反约束的放置

            # 记录有效放置
            level_info['used_length'] += effective_length
            level_info['used_volume'] += box.volume
            level_info['boxes'].append({
                'box': box,
                'orientation': placement.orientation,
                'effective_length': effective_length,
                'effective_width': effective_width,
                'unique_id': placement.box_id,
                'actual_length': box.length,
                'actual_width': box.width,
                'height': box.height,
                'safety_distance': self.safety_distance,
                'pallet_length': box.pallet_length,
                'pallet_width': box.pallet_width
            })
            used_boxes[original_box_id] = used_boxes.get(original_box_id, 0) + 1
            total_used_volume += box.volume

        # 计算体积利用率
        volume_utilization = total_used_volume / total_available_volume if total_available_volume > 0 else 0

        # 对每个货架层的箱子按体积从大到小排序
        for key in shelf_usage:
            shelf_usage[key]['boxes'] = sorted(
                shelf_usage[key]['boxes'],
                key=lambda x: x['box'].volume,
                reverse=True
            )

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
                    # 根据朝向获取箱子的实际尺寸（不使用max函数）
                    if box_info['orientation'] == 0:  # 长边朝外
                        box_length = box_info['box'].length
                        box_width = box_info['box'].width
                    else:  # 宽边朝外
                        box_length = box_info['box'].width
                        box_width = box_info['box'].length

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
                        text=[f"{box_info['box'].id}<br>{box_info['box'].length}×{box_info['box'].width}×{box_info['box'].height}"],
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
                    if box_info['orientation'] == 0:  # 长边朝外
                        actual_length = box_info['box'].length
                        actual_width = box_info['box'].width
                    else:  # 宽边朝外
                        actual_length = box_info['box'].width
                        actual_width = box_info['box'].length

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
    pop_size = 100
    generations = 5000
    crossover_rate = 0.9
    mutation_rate = 0.1
    elite_size = 5
    safety_distance = 0.03  # 3公分安全距离

    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx"], help="请上传包含箱子尺寸和库存数据的Excel文件")

    # 初始化session state
    if 'optimized' not in st.session_state:
        st.session_state.optimized = False
    if 'best_solution' not in st.session_state:
        st.session_state.best_solution = None
    if 'shelf_usage' not in st.session_state:
        st.session_state.shelf_usage = None
    if 'ga' not in st.session_state:
        st.session_state.ga = None

    if uploaded_file is not None:
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # 显示数据预览
            st.subheader("数据预览")
            df = pd.read_excel(tmp_file_path)
            st.dataframe(df.head())

            if st.button("🚀 开始优化", type="primary") or st.session_state.optimized:
                if not st.session_state.optimized:
                    with st.spinner("正在运行遗传算法优化..."):
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
                        shelf_usage, used_boxes, total_used_volume, volume_utilization = ga.decode_solution(
                            best_solution)

                        # 保存结果到session state
                        st.session_state.optimized = True
                        st.session_state.best_solution = best_solution
                        st.session_state.shelf_usage = shelf_usage
                        st.session_state.used_boxes = used_boxes
                        st.session_state.total_used_volume = total_used_volume
                        st.session_state.volume_utilization = volume_utilization
                        st.session_state.ga = ga
                else:
                    # 从session state获取结果
                    ga = st.session_state.ga
                    shelf_usage = st.session_state.shelf_usage
                    used_boxes = st.session_state.used_boxes
                    total_used_volume = st.session_state.total_used_volume
                    volume_utilization = st.session_state.volume_utilization

                # 显示结果
                st.success("优化完成！")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("体积利用率", f"{volume_utilization:.2%}")
                with col2:
                    st.metric("使用体积", f"{total_used_volume:.2f} m³")
                with col3:
                    total_available_volume = sum(
                        shelf.length * shelf.width * shelf.height * shelf.levels for shelf in ga.shelves)
                    st.metric("总可用体积", f"{total_available_volume:.2f} m³")

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

                # 详细放置方案表格
                st.subheader("详细放置方案")

                placement_data = []
                for (shelf_idx, level), info in shelf_usage.items():
                    for box_info in info['boxes']:
                        placement_data.append({
                            "货架组": shelf_idx + 1,
                            "层级": level + 1,
                            "箱子ID": box_info['box'].id,
                            "实际尺寸": f"{box_info['actual_length']}×{box_info['actual_width']}×{box_info['height']}m",
                            "体积": f"{box_info['box'].volume:.2f} m³",
                            "朝向": "长边朝外",
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
