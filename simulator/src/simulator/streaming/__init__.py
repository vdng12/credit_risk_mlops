"""Streaming (online lane) machinery.

The generator writes postgres OLTP and Debezium CDC re-publishes it; the
online lane's *delivery* is re-paced day by day here (SimClock) with the
streaming problems (burst, duplicate replay, late arrival) attached before
the write, so the CDC stream shows the corruption the rubric wants.
"""
