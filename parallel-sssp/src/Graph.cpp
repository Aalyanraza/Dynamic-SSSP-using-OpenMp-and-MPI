// Graph.cpp
#include "Graph.h"
#include <algorithm> 

Graph::Graph(bool directed) : directed(directed) {}

void Graph::addEdge(Node u, Node v, Weight w) {
    adjList[u].emplace_back(v, w);
    if (!directed) {
        adjList[v].emplace_back(u, w);
    }
}

void Graph::removeEdge(Node u, Node v) {
    auto& edges = adjList[u];
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                [v](const Edge& e) { return e.first == v; }), edges.end());
    if (!directed) {
        auto& edgesRev = adjList[v];
        edgesRev.erase(std::remove_if(edgesRev.begin(), edgesRev.end(),
                [u](const Edge& e) { return e.first == u; }), edgesRev.end());
    }
}

void Graph::updateWeight(Node u, Node v, Weight w) {
    for (auto& e : adjList[u]) {
        if (e.first == v) {
            e.second = w;
            break;
        }
    }
    if (!directed) {
        for (auto& e : adjList[v]) {
            if (e.first == u) {
                e.second = w;
                break;
            }
        }
    }
}

std::vector<Graph::Edge> Graph::getNeighbors(Node u) const {
    if (adjList.find(u) != adjList.end()) {
        return adjList.at(u);
    }
    return {};
}

void Graph::printGraph() const {
    for (const auto& [node, neighbors] : adjList) {
        std::cout << node << ": ";
        for (const auto& [dest, weight] : neighbors) {
            std::cout << "(" << dest << ", " << weight << ") ";
        }
        std::cout << "\n";
    }
}

const std::unordered_map<Graph::Node, std::vector<Graph::Edge>>& Graph::getAdjList() const {
    return adjList;
}
