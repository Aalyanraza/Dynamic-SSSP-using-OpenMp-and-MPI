// Dijkstra.h
#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include "Graph.h"
#include <vector>
#include <queue>
#include <limits>
#include <fstream>

void dijkstraSSSP(const Graph& g, Graph::Node source, const std::string& outputFile) {
    std::unordered_map<Graph::Node, double> dist;
    const auto& adj = g.getAdjList();

    for (const auto& [node, _] : adj) {
        dist[node] = std::numeric_limits<double>::infinity();
    }

    dist[source] = 0.0;

    using PQNode = std::pair<double, Graph::Node>;
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<>> pq;
    pq.emplace(0.0, source);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        // Check all neighbors in both directions
        for (const auto& [v, w] : g.getNeighbors(u)) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.emplace(dist[v], v);
            }
        }

        // Optional: for a robust undirected behavior, look for reverse edges too
        for (const auto& [v, w] : adj) {
            for (const auto& [vv, ww] : w) {
                if (vv == u) {
                    if (dist[v] + ww < dist[u]) {
                        dist[u] = dist[v] + ww;
                        pq.emplace(dist[u], u);
                    }
                }
            }
        }
    }

    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << "\n";
        return;
    }

    out << "# SSSP from source node " << source << "\n";
    for (const auto& [node, d] : dist) {
        out << node << " " << d << "\n";
    }

    out.close();
    std::cout << "SSSP results written to '" << outputFile << "'\n";
}

#endif // DIJKSTRA_H
