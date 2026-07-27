# adapted from openpi
from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FieldMapping:
    field: str
    out_key: str


@dataclass
class TopicConfig:
    topic: str
    type: str
    adapter: str
    mappings: List[FieldMapping]


@dataclass
class YAMLConfig:
    protocol: str
    topics: List[TopicConfig]


class YAMLConfigLoader:
    """
    YAML configuration loader
    Example structure:
    protocol: ros2
    topics:
      - topic: /joint_states
        type: sensor_msgs.msg.JointState
        adapter: GenericAdapter
        mappings:
          - field: position
            out_key: joint_positions
          - field: velocity
            out_key: joint_velocities
      - topic: /wrench_topic
        type: geometry_msgs.msg.Wrench
        adapter: WrenchAdapter
        mappings:
          - field: force
            out_key: wrench_force
    """

    def __init__(self, path: str):
        self.path = path
        self.config: Optional[YAMLConfig] = None
        self.load()

    def load(self) -> YAMLConfig:
        """
        Read the YAML file and parse it into a YAMLConfig structure
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty YAML configuration at {self.path}")

        protocol = data.get('protocol')
        topics_raw = data.get('topics', [])

        topics: List[TopicConfig] = []
        for t in topics_raw:
            topic_name = t.get('topic')
            msg_type = t.get('type')
            adapter = t.get('adapter', 'GenericAdapter')  # use the generic adapter by default
            mappings_raw = t.get('mappings', [])

            mappings: List[FieldMapping] = []
            for m in mappings_raw:
                fld = m.get('field')
                out_key = m.get('out_key')
                if fld is None or out_key is None:
                    print(f"Invalid mapping: {m}")
                    continue
                mappings.append(FieldMapping(field=fld, out_key=out_key))

            topics.append(TopicConfig(
                topic=topic_name, 
                type=msg_type, 
                adapter=adapter,
                mappings=mappings
            ))

        self.config = YAMLConfig(protocol=protocol, topics=topics)
        return self.config

    def get_topics(self) -> List[TopicConfig]:
        """
        Get the loaded list of Topic configs; raise if not loaded yet
        """
        if self.config is None:
            raise RuntimeError("YAMLConfigLoader: configuration is not loaded yet. Call load() first.")
        return self.config.topics

    def get_protocol(self) -> str:
        """
        Get the protocol field of the current YAML configuration
        """
        if self.config is None:
            raise RuntimeError("YAMLConfigLoader: configuration is not loaded yet. Call load() first.")
        return self.config.protocol

    def find_topic(self, topic: str) -> Optional[TopicConfig]:
        """
        Find the TopicConfig matching the given topic name in the loaded configuration
        """
        if self.config is None:
            raise RuntimeError("YAMLConfigLoader: configuration is not loaded yet. Call load() first.")
        for t in self.config.topics:
            if t.topic == topic:
                return t
        return None

    def has_topic(self, topic: str) -> bool:
        """
        Check whether the configuration contains the given topic
        """
        return self.find_topic(topic) is not None

    def get_type_by_topic(self, topic: str) -> str:
        """
        Get the message type name for the given topic
        Raises KeyError if the topic does not exist
        """
        t = self.find_topic(topic)
        if t is None:
            raise KeyError(f"Topic '{topic}' not found in YAML configuration.")
        return t.type

    def get_adapter_by_topic(self, topic: str) -> str:
        """
        Get the adapter name for the given topic
        Raises KeyError if the topic does not exist
        """
        t = self.find_topic(topic)
        if t is None:
            raise KeyError(f"Topic '{topic}' not found in YAML configuration.")
        return t.adapter

    def get_mappings_by_topic(self, topic: str) -> List[FieldMapping]:
        """
        Get the field mappings (list of FieldMapping) for the given topic
        Raises KeyError if the topic does not exist
        """
        t = self.find_topic(topic)
        if t is None:
            raise KeyError(f"Topic '{topic}' not found in YAML configuration.")
        return t.mappings

    def __repr__(self) -> str:
        return f"YAMLConfigLoader(path={self.path!r}, loaded={self.config is not None})"
