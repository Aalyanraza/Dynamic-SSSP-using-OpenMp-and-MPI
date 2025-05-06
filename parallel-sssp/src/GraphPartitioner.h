#ifndef GRAPH_PARTITIONER_H
#define GRAPH_PARTITIONER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <tuple>
#include <metis.h>

bool partitionGraphMETIS_fromFile(
    const std::string& filename,
    int numParts,
    std::unordered_map<int, int>& nodeToPartition
) {
    using idx_t = int;

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return false;
    }

    std::set<idx_t> all_nodes;
    std::vector<std::tuple<idx_t, idx_t, idx_t>> edges;

    idx_t u, v, w;
    while (infile >> u >> v >> w) {
        if (u != v) {
            edges.emplace_back(u, v, w);
            edges.emplace_back(v, u, w);  // for undirected
        }
        all_nodes.insert(u);
        all_nodes.insert(v);
    }

    infile.close();

    idx_t numVertices = *all_nodes.rbegin() + 1;  // assuming 0-based indexing
    std::vector<std::vector<std::pair<idx_t, idx_t>>> adjList(numVertices);

    for (auto [src, dest, weight] : edges) {
        adjList[src].emplace_back(dest, weight);
    }

    std::vector<idx_t> xadj, adjncy, adjwgt;
    xadj.push_back(0);
    for (idx_t i = 0; i < numVertices; ++i) {
        for (const auto& [neighbor, weight] : adjList[i]) {
            adjncy.push_back(neighbor);
            adjwgt.push_back(weight);
        }
        xadj.push_back(adjncy.size());
    }

    std::vector<idx_t> part(numVertices);
    idx_t objval;
    idx_t ncon = 1;

    int status = METIS_PartGraphKway(
        &numVertices, &ncon,
        xadj.data(), adjncy.data(),
        nullptr, nullptr, adjwgt.data(),
        &numParts, nullptr, nullptr,
        nullptr, &objval, part.data()
    );

    if (status != METIS_OK) {
        std::cerr << "METIS_PartGraphKway failed with status: " << status << "\n";
        return false;
    }

    for (idx_t i = 0; i < numVertices; ++i) {
        nodeToPartition[i] = part[i];
    }

    std::cout << "Graph partitioned into " << numParts << " parts (Edge cut = " << objval << ").\n";
    return true;
}

#endif // GRAPH_PARTITIONER_H
