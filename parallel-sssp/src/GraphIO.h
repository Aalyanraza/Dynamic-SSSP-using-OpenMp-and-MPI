// GraphIO.h
#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include "Graph.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool readGraphFromFile(Graph& g, const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return false;

    int u, v, w;
    while (in >> u >> v >> w) {
        g.addEdge(u, v, w);  // Add only as given in file
    }

    return true;
}


#endif // GRAPH_IO_H
