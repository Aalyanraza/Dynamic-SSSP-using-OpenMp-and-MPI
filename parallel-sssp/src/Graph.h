// Graph.h
#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>
#include <utility>
#include <limits>
#include <iostream>

class Graph {
public:
    using Node = int;
    using Weight = double;
    using Edge = std::pair<Node, Weight>;

    Graph(bool directed = false);

    void addEdge(Node u, Node v, Weight w);
    void removeEdge(Node u, Node v);
    void updateWeight(Node u, Node v, Weight w);
    std::vector<Edge> getNeighbors(Node u) const;
    void printGraph() const;

    const std::unordered_map<Node, std::vector<Edge>>& getAdjList() const;

private:
    std::unordered_map<Node, std::vector<Edge>> adjList;
    bool directed;
};

#endif // GRAPH_H
