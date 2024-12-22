#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" このモジュールは、ランダムなウェイポイントに従ってトラックを走行し、他の車両を回避するエージェントを実装します。
エージェントは信号にも対応します。 """

from enum import Enum

import carla
import os.path as osp
import numpy as np
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle, compute_yaw_difference

from skimage.io import imread


WORLD_OFFSETS = {
    'Town01' : (-52.059906005859375, -52.04995942115784),
    'Town02' : (-57.459808349609375, 55.3907470703125)
}
PIXELS_PER_METER = 5

class AgentState(Enum):
    """
    AGENT_STATE はローミングエージェントの可能な状態を表します
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class Agent(object):
    """
     CARLA内でエージェントを定義するための基底クラス
    """

    def __init__(self, vehicle):
        """
        コンストラクタ

        :param vehicle: ローカルプランナーロジックを適用する対象のアクター
        """
        self._vehicle = vehicle
        self._last_traffic_light = None

        self._world = None
        self._map = None
        self._debug_info = None

        while self._world is None and self._map is None:
            try:
                self._world = self._vehicle.get_world()
                self._map = self._world.get_map()
            except RuntimeError as e:
                print(e)

        self._road_map = imread(osp.join(osp.dirname(__file__), '%s.png' % self._map.name))

    def run_step(self, inputs=None, debug=False):
        """
        ナビゲーションの1ステップを実行します。
        :return: 車両制御情報
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def _is_light_red(self, lights_list):
        """
        赤信号が影響しているかを確認するメソッド。このバージョンはヨーロッパおよびアメリカの信号スタイルに対応しています。

        :param lights_list: TrafficLightオブジェクトを含むリスト
        :return: (bool値, traffic_light) のタプル
                 - bool値は赤信号が影響している場合True、そうでない場合False
                 - traffic_light は影響している信号のオブジェクト、ない場合はNone
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        ヨーロッパスタイルの信号を確認するためのメソッド。

        :param lights_list: TrafficLightオブジェクトを含むリスト
        :return: (bool値, traffic_light) のタプル
                 - bool値は赤信号が影響している場合True、そうでない場合False
                 - traffic_light は影響している信号のオブジェクト、ない場合はNone


        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        
        for traffic_light in lights_list:
            location = traffic_light.get_location()
            object_waypoint = self._map.get_waypoint(location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            if not is_within_distance_ahead(
                    location,
                    ego_vehicle_location,
                    self._vehicle.get_transform().rotation.yaw,
                    self._proximity_threshold, degree=60):
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        アメリカスタイルの信号を確認するためのメソッド。

        :param lights_list: TrafficLightオブジェクトを含むリスト
        :return: (bool値, traffic_light) のタプル
                 - bool値は赤信号が影響している場合True、そうでない場合False
                 - traffic_light は影響している信号のオブジェクト、ない場合はNone
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # 自社料が交差点に侵入している場合早く止まる
            return (False, None)

        # targetが交差点の場合、信号を確認する
        if self._local_planner._target_waypoint is not None:
            if self._local_planner._target_waypoint.is_intersection:
                potential_lights = []
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    # 信号機の位置、自車の位置、自車の向きを使って距離と角度を計算
                    magnitude, angle = compute_magnitude_angle(
                            loc,
                            ego_vehicle_location,
                            self._vehicle.get_transform().rotation.yaw)
                    # 距離が80m未満、角度が25度未満の信号を選択
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print(
                                '=== Magnitude = {} | Angle = {} | ID = {}'.format(
                                    sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        # 以前に判定した信号がない場合、選択した信号を保存
                        self._last_traffic_light = sel_traffic_light

                    # 指定した信号機が赤の場合、Trueを返す
                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)

    def _is_walker_hazard(self, walkers_list):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for walker in walkers_list:
            # 歩行者の位置を取得して、自車との距離を計算
            loc = walker.get_location()
            dist = loc.distance(ego_vehicle_location)
            # 歩行者との距離に応じて、注意範囲の角度を計算
            # clipで最小値を1.5m、最大値を10.5mに制限
            # 162 / (dist + 0.3) で角度を計算しているので、距離が近いほど角度が大きくなる
            degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)
            # ほこうしゃが歩道上にいる場合は無視
            if self._is_point_on_sidewalk(loc):
                continue

            # 歩行者が自車の前方にいる場合、Trueを返す
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold, degree=degree):
                return (True, walker)

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        指定された車両が進行方向の障害物であるかを確認します。このために、対象の車両が走行している道路と車線を考慮し、幾何学的なテストを実行して、対象車両が自車両の一定距離内にいるかどうかを確認します。

        注意: このメソッドは近似値を使用しており、非常に大きな車両では失敗する可能性があります。具体的には、車両の中心が別の車線上にある場合でも、その車両の一部が自車両の車線内に入るケースが該当します。

        :param vehicle_list: 障害物として確認する対象車両のリスト :return: (bool_flag, vehicle) というタプル - bool_flag は、進行方向に車両がいて進行を妨げている場合に True を返し、それ以外の場合は False を返します。 - vehicle は障害物となっている車両オブジェクトそのものを返します。 
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_orientation = self._vehicle.get_transform().rotation.yaw
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            loc = target_vehicle.get_location()
            ori = target_vehicle.get_transform().rotation.yaw

            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())

            # waiting = ego_vehicle_waypoint.is_intersection and target_vehicle.get_traffic_light_state() == carla.TrafficLightState.Red
            # print ("Not our lane: ", other_lane)
            # print ("Waiting", waiting)

            # if the object is not in our lane it's not an obstacle
            # if not ego_vehicle_waypoint.is_intersection and target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            #     continue

            # if the object is waiting for it's not an obstacle
            # if waiting:
            #     continue

            # もし対向者と自社の角度が150ド以下で、車両の中心として、左右に22.5ド以下かつ一定距離内にいる車両を障害物として扱う
            if compute_yaw_difference(ego_vehicle_orientation, ori) <= 150 and is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold, degree=45):
                return (True, target_vehicle)

        return (False, None)


    def emergency_stop(self):
        """
        緊急停止コマンドを車両に送信します。
        :return: 車両制御情報
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control


    def _world_to_pixel(self, location, offset=(0, 0)):
        """
        ワールド座標をピクセル座標に変換します。
        """
        if self._map.name in WORLD_OFFSETS:
            # 使用しているマップに応じたオフセットを取得
            # Town01 の基準座標は (-52.059906, -52.049959)。
            # Town02 の基準座標は (-57.459808, 55.390747)。
            world_offset = WORLD_OFFSETS[self._map.name]
            # ワールド座標(location)から基準座標を引き、ローカル座標に変換し、ピクセル座標に変換
            x = PIXELS_PER_METER * (location.x - world_offset[0])
            y = PIXELS_PER_METER * (location.y - world_offset[1])
            return [int(x - offset[0]), int(y - offset[1])]
        else:
            return [0, 0]
        # world_offset = WORLD_OFFSETS[self._map.name]
        # x = PIXELS_PER_METER * (location.x - world_offset[0])
        # y = PIXELS_PER_METER * (location.y - world_offset[1])
        # return [int(x - offset[0]), int(y - offset[1])]
    
    
    def _is_point_on_sidewalk(self, loc):
        """
        指定された地点が歩道上かどうかを判定します。

        :param loc: 場所オブジェクト
        :return: 真偽値
        """
        # Convert to pixel coordinate
        # ワールド座標をピクセル座標に変換
        pixel_x, pixel_y = self._world_to_pixel(loc)
        # 座標が範囲外の場合、制限
        pixel_y = np.clip(pixel_y, 0, self._road_map.shape[0]-1)
        pixel_x = np.clip(pixel_x, 0, self._road_map.shape[1]-1)
        # map画像のpixel座標から、その地点のピクセル値を取得
        point = self._road_map[pixel_y, pixel_x, 0]
        # その地点のピクセル値が0の場合、歩道上にあると判断
        return point == 0
