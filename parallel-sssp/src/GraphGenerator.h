// GraphGenerator.h
#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include "Graph.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>

void generateSparseDirectedGraph(Graph& g, int numNodes, const std::string& outputFile, int avgOutDegree = 3) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::set<std::pair<int, int>> addedEdges;
    int edgeCount = 0;

    for (int u = 0; u < numNodes; ++u) {
        std::set<int> targets;

        while ((int)targets.size() < avgOutDegree) {
            int v = std::rand() % numNodes;
            if (v != u && addedEdges.find({u, v}) == addedEdges.end()) {
                targets.insert(v);
                addedEdges.insert({u, v});
                g.addEdge(u, v, 1.0);
                edgeCount++;
            }
        }
    }

    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << outputFile << " for writing.\n";
        return;
    }

    out << "# " << numNodes << " " << edgeCount << "\n";
    for (const auto& [from, neighbors] : g.getAdjList()) {
        for (const auto& [to, weight] : neighbors) {
            out << from << " " << to << " " << weight << "\n";
        }
    }

    out.close();
    std::cout << "Sparse graph with avg out-degree " << avgOutDegree
              << " written to '" << outputFile << "'.\n";
}

#endif // GRAPH_GENERATOR_H
