#include "Graph.h"
#include "GraphIO.h"
#include "GraphPartitioner.h"
#include "Dijkstra.h"
#include <mpi.h>
#include <omp.h>
#include <set>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <limits>
#include <algorithm>

const int INF = std::numeric_limits<int>::max();

// Serialize the partition as a flat list: [u, v, w, u, v, w, ...]
void serializePartition(
    const Graph& g, 
    const std::vector<int>& nodes, 
    const std::unordered_map<int, int>& nodeToPartition,
    int currentPartition,
    std::vector<int>& flatData
) {
    const auto& adj = g.getAdjList();
    std::set<std::pair<int, int>> addedEdges;

    for (int u : nodes) {
        for (const auto& [v, w] : adj.at(u)) {
            // Include all edges where u belongs to this partition
            if (nodeToPartition.at(u) == currentPartition) {
                auto edge = std::make_pair(std::min(u, v), std::max(u, v));
                if (addedEdges.count(edge) == 0) {
                    flatData.push_back(u);
                    flatData.push_back(v);
                    flatData.push_back(w);
                    addedEdges.insert(edge);
                }
            }
        }
    }
}

std::set<int> findGhostNodes(
    const std::vector<int>& ownedNodes,
    const Graph& fullGraph,
    const std::unordered_map<int, int>& nodeToPartition,
    int currentPartition
) {
    const auto& adj = fullGraph.getAdjList();
    std::set<int> ghosts;

    for (int u : ownedNodes) {
        auto it = adj.find(u);
        if (it == adj.end()) continue;

        for (const auto& [v, _] : it->second) {
            if (nodeToPartition.at(v) != currentPartition) {
                ghosts.insert(v);
            }
        }
    }

    return ghosts;
}

// Distribute ghost node information from master to workers
void distributeGhostNodes(
    const Graph& fullGraph,
    const std::unordered_map<int, int>& nodeToPartition,
    const std::vector<std::vector<int>>& partNodes,
    int numParts
) {
    std::vector<std::set<int>> ghostsByPartition(numParts);
    
    // Identify ghost nodes for each partition
    for (int p = 0; p < numParts; ++p) {
        ghostsByPartition[p] = findGhostNodes(partNodes[p], fullGraph, nodeToPartition, p);
    }
    
    // Master sends ghost node information to workers
    if (numParts > 1) {
        for (int worker = 1; worker < numParts; ++worker) {
            // Convert set to vector for MPI
            std::vector<int> ghostNodes(ghostsByPartition[worker].begin(), 
                                        ghostsByPartition[worker].end());
            int numGhosts = ghostNodes.size();
            
            // Send ghost node count
            MPI_Send(&numGhosts, 1, MPI_INT, worker, 2, MPI_COMM_WORLD);
            
            // Send ghost nodes if any exist
            if (numGhosts > 0) {
                MPI_Send(ghostNodes.data(), numGhosts, MPI_INT, worker, 3, MPI_COMM_WORLD);
            }
        }
    }
}

// Structure to hold node distance data for communication
struct NodeDistance {
    int node;
    int distance;
    int parent;
};

// Process CE (Concurrent Edge) Changes - Algorithm 2 from the paper
void processCE(
    Graph& localGraph,
    const std::set<std::pair<int, int>>& T,  // SSSP tree edges
    const std::vector<std::pair<int, int>>& deletedEdges,
    const std::vector<std::tuple<int, int, int>>& insertedEdges,  // <u, v, w>
    std::vector<int>& dist,
    std::vector<int>& parent,
    std::vector<bool>& affected_del,
    std::vector<bool>& affected
) {
    // Create a temporary graph for processing
    Graph Gu = localGraph;
    
    // Process deleted edges
    #pragma omp parallel for
    for (size_t i = 0; i < deletedEdges.size(); i++) {
        int u = deletedEdges[i].first;
        int v = deletedEdges[i].second;
        
        // Check if edge is part of SSSP tree
        if (T.count(std::make_pair(u, v)) > 0 || T.count(std::make_pair(v, u)) > 0) {
            // Find y (the vertex with greater distance)
            int y = (dist[u] > dist[v]) ? u : v;
            
            #pragma omp critical
            {
                // Mark y as affected by deletion
                dist[y] = INF;
                affected_del[y] = true;
                affected[y] = true;
            }
        }
        
        // Mark edge as deleted (remove from graph)
        Gu.removeEdge(u, v);
    }
    
    // Process inserted edges
    #pragma omp parallel for
    for (size_t i = 0; i < insertedEdges.size(); i++) {
        int u = std::get<0>(insertedEdges[i]);
        int v = std::get<1>(insertedEdges[i]);
        int w = std::get<2>(insertedEdges[i]);
        
        // Ensure u has the smaller distance (as per algorithm)
        if (dist[u] > dist[v]) {
            std::swap(u, v);
        }
        
        // Check relaxation condition
        if (dist[v] > dist[u] + w) {
            #pragma omp critical
            {
                // Update distance and parent
                dist[v] = dist[u] + w;
                parent[v] = u;
                affected[v] = true;
                
                // Add edge to Gu
                Gu.addEdge(u, v, w);
            }
        }
    }
    
    // Replace local graph with updated graph
    localGraph = Gu;
}

// Update affected vertices - Algorithm 3 from the paper
void updateAffectedVertices(
    Graph& localGraph,
    const std::set<std::pair<int, int>>& T,  // SSSP tree edges
    std::vector<int>& dist,
    std::vector<int>& parent,
    std::vector<bool>& affected_del,
    std::vector<bool>& affected,
    const std::set<int>& localNodes
) {
    // First while loop - handle deletions
    bool has_affected_del = true;
    while (has_affected_del) {
        has_affected_del = false;
        
        std::vector<int> affected_del_nodes;
        for (int v : localNodes) {
            if (affected_del[v]) {
                affected_del_nodes.push_back(v);
            }
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < affected_del_nodes.size(); i++) {
            int v = affected_del_nodes[i];
            
            #pragma omp critical
            {
                affected_del[v] = false;
            }
            
            // Find children of v in SSSP tree
            std::vector<int> children;
            for (int node : localNodes) {
                if (parent[node] == v) {
                    children.push_back(node);
                }
            }
            
            for (int c : children) {
                #pragma omp critical
                {
                    dist[c] = INF;
                    affected_del[c] = true;
                    affected[c] = true;
                }
            }
            
            #pragma omp critical
            {
                has_affected_del = has_affected_del || !children.empty();
            }
        }
    }
    
    // Second while loop - handle relaxations
    bool has_affected = true;
    while (has_affected) {
        has_affected = false;
        
        std::vector<int> affected_nodes;
        for (int v : localNodes) {
            if (affected[v]) {
                affected_nodes.push_back(v);
            }
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < affected_nodes.size(); i++) {
            int v = affected_nodes[i];
            
            #pragma omp critical
            {
                affected[v] = false;
            }
            
            // Process neighbors
            const auto& AdjList = localGraph.getAdjList();
            if (AdjList.find(v) != AdjList.end()) {
                for (const auto& [n, w] : AdjList.at(v)) {
                    if (dist[n] > dist[v] + w) {
                        #pragma omp critical
                        {
                            dist[n] = dist[v] + w;
                            parent[n] = v;
                            affected[n] = true;
                            has_affected = true;
                        }
                    } else if (dist[v] > dist[n] + w) {
                        #pragma omp critical
                        {
                            dist[v] = dist[n] + w;
                            parent[v] = n;
                            affected[v] = true;
                            has_affected = true;
                        }
                    }
                }
            }
        }
    }
}

// Exchange boundary data between partitions
void exchangeBoundaryData(
    const std::vector<int>& ghostNodes,
    std::vector<int>& dist,
    std::vector<int>& parent,
    int rank,
    int size,
    const std::unordered_map<int, int>& nodeToPartition,
    const std::set<int>& localNodes
) {
    const int MAX_NODES = 10000;  // Adjust based on expected graph size
    
    // Prepare send buffer for each process
    std::vector<std::vector<NodeDistance>> sendBuffers(size);
    
    // Prepare data to send
    for (int v : localNodes) {
        if (dist[v] != INF) {  // Only send nodes with valid distances
            for (int target = 0; target < size; target++) {
                if (target != rank) {
                    NodeDistance nd;
                    nd.node = v;
                    nd.distance = dist[v];
                    nd.parent = parent[v];
                    sendBuffers[target].push_back(nd);
                }
            }
        }
    }
    
    // Send and receive buffer sizes
    std::vector<int> sendCounts(size, 0);
    std::vector<int> recvCounts(size, 0);
    
    for (int i = 0; i < size; i++) {
        sendCounts[i] = sendBuffers[i].size();
    }
    
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare flat send buffers
    std::vector<NodeDistance> allSendData;
    std::vector<int> sendDisplacements(size, 0);
    int totalSend = 0;
    
    for (int i = 0; i < size; i++) {
        sendDisplacements[i] = totalSend;
        allSendData.insert(allSendData.end(), sendBuffers[i].begin(), sendBuffers[i].end());
        totalSend += sendCounts[i];
    }
    
    // Calculate receive displacements
    std::vector<int> recvDisplacements(size, 0);
    int totalRecv = 0;
    
    for (int i = 0; i < size; i++) {
        recvDisplacements[i] = totalRecv;
        totalRecv += recvCounts[i];
    }
    
    // Create MPI datatype for NodeDistance
    MPI_Datatype nodeDist_type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    
    // Calculate displacements
    NodeDistance temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.node, &displacements[0]);
    MPI_Get_address(&temp.distance, &displacements[1]);
    MPI_Get_address(&temp.parent, &displacements[2]);
    
    // Make relative to base_address
    for (int i = 0; i < 3; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);
    }
    
    MPI_Type_create_struct(3, blocklengths, displacements, types, &nodeDist_type);
    MPI_Type_commit(&nodeDist_type);
    
    // Allocate receive buffer
    std::vector<NodeDistance> recvBuffer(totalRecv);
    
    // Exchange data
    MPI_Alltoallv(
        allSendData.data(), sendCounts.data(), sendDisplacements.data(), nodeDist_type,
        recvBuffer.data(), recvCounts.data(), recvDisplacements.data(), nodeDist_type,
        MPI_COMM_WORLD
    );
    
    // Update local data with received ghost node information
    #pragma omp parallel for
    for (int i = 0; i < totalRecv; i++) 
    {
        NodeDistance& nd = recvBuffer[i];
        int node = nd.node;
        
        // Only update if node is a ghost node we care about
        if (ghostNodes.empty() || std::binary_search(ghostNodes.begin(), ghostNodes.end(), node)) 
        {
            #pragma omp critical
            {
                // Update if new distance is better
                if (dist[node] > nd.distance) {
                    dist[node] = nd.distance;
                    parent[node] = nd.parent;
                }
            }
        }
    }
    
    // Clean up
    MPI_Type_free(&nodeDist_type);
}

// Build SSSP tree edges set
std::set<std::pair<int, int>> buildSSSPTree(
    const std::vector<int>& parent,
    const std::set<int>& localNodes
) {
    std::set<std::pair<int, int>> treeEdges;
    
    for (int v : localNodes) {
        if (parent[v] != -1) {  // Not source or unreachable
            int u = parent[v];
            treeEdges.insert(std::make_pair(std::min(u, v), std::max(u, v)));
        }
    }
    
    return treeEdges;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) 
    {
        if (rank == 0)
            std::cerr << "Error: Run with at least 2 processes (1 master + >=1 workers).\n";
        MPI_Finalize();
        return 1;
    }

    // Set number of OpenMP threads
    int num_threads = 4;  // Adjust as needed
    omp_set_num_threads(num_threads);
    
    const int sourceNode = 0;  // Source node for SSSP
    
    if (rank == 0) 
    {
        // === MASTER ===
        Graph g(false);
        if (!readGraphFromFile(g, "graph_data.txt")) 
        {
            std::cerr << "Error reading graph.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Get maximum node ID to size arrays correctly
        int maxNodeId = 0;
        const auto& adjList1 = g.getAdjList();
        for (const auto& [node, _] : adjList1) {
            maxNodeId = std::max(maxNodeId, node);
            for (const auto& [neighbor, _] : adjList1.at(node)) {
                maxNodeId = std::max(maxNodeId, neighbor);
            }
        }
        
        // Use METIS partitioning from file
        std::unordered_map<int, int> nodeToPartition;
        int numParts = size;
        if (!partitionGraphMETIS_fromFile("graph_data.txt", numParts, nodeToPartition)) {
            std::cerr << "Partitioning failed.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Group nodes by partition
        std::vector<std::vector<int>> partNodes(numParts);
        for (const auto& [node, part] : nodeToPartition) {
            if (part >= 0 && part < numParts) {
                partNodes[part].push_back(node);
            }
        }

        std::ifstream updateFile("updates.txt");
        if (!updateFile.is_open()) {
            std::cerr << "Failed to open updates.txt\n";
            return 1;
        }

        std::string type;
        int u, v, w;

        // Simulate edge changes - for demonstration, let's delete and insert edges
        std::vector<std::pair<int, int>> deletedEdges;

        std::vector<std::tuple<int, int, int>> insertedEdges;

        while (updateFile >> type >> u >> v >> w) 
        {
            if (type == "I") 
            {
                insertedEdges.emplace_back(u, v, w);
            } 
            else if (type == "D") 
            {
                deletedEdges.emplace_back(u, v);
            }
        }
        updateFile.close();


        // Distribute ghost nodes
        distributeGhostNodes(g, nodeToPartition, partNodes, numParts);

        // Print ghost nodes for each partition
        for (int p = 0; p < numParts; ++p) {
            std::set<int> ghosts = findGhostNodes(partNodes[p], g, nodeToPartition, p);
            std::cout << "Ghost nodes for partition " << p << ": ";
            for (int ghost : ghosts) std::cout << ghost << " ";
            std::cout << "\n";
        }

        // Send graph partitions to workers
        for (int worker = 1; worker < size; ++worker) {
            std::vector<int> flat;
            serializePartition(g, partNodes[worker], nodeToPartition, worker, flat);

            int dataSize = flat.size();
            MPI_Send(&dataSize, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            if (dataSize > 0) {
                MPI_Send(flat.data(), dataSize, MPI_INT, worker, 1, MPI_COMM_WORLD);
            }
            
            // Send max node ID
            MPI_Send(&maxNodeId, 1, MPI_INT, worker, 4, MPI_COMM_WORLD);
        }

        // Master processes its own part
        std::vector<int> masterFlat;
        serializePartition(g, partNodes[0], nodeToPartition, 0, masterFlat);

        std::cout << "Master (rank 0) has " << (masterFlat.size() / 3) << " edges\n";

        // Create master's local graph
        Graph localGraph(false);
        for (size_t i = 0; i < masterFlat.size(); i += 3) 
        {
            int u = masterFlat[i];
            int v = masterFlat[i + 1];
            int w = masterFlat[i + 2];
            localGraph.addEdge(u, v, w);
        }
        
        // Initialize distance and parent arrays
        std::vector<int> dist(maxNodeId + 1, INF);
        std::vector<int> parent(maxNodeId + 1, -1);
        std::vector<bool> affected_del(maxNodeId + 1, false);
        std::vector<bool> affected(maxNodeId + 1, false);
        
        // Set source distance
        dist[sourceNode] = 0;
        
        // Convert partition nodes to set for faster lookups
        std::set<int> localNodes(partNodes[0].begin(), partNodes[0].end());
        
        // Get ghost nodes for master
        std::set<int> masterGhosts = findGhostNodes(partNodes[0], g, nodeToPartition, 0);
        std::vector<int> ghostNodes(masterGhosts.begin(), masterGhosts.end());
        
        // Initial relaxation from source node
        const auto& adjList = localGraph.getAdjList();
        if (adjList.find(sourceNode) != adjList.end()) 
        {
            for (const auto& [neighbor, weight] : adjList.at(sourceNode)) 
            {
                dist[neighbor] = weight;
                parent[neighbor] = sourceNode;
                affected[neighbor] = true;
            }
        }
        // Initialize SSSP tree edges
        std::set<std::pair<int, int>> T;
        


        // Main SSSP computation loop
        int iterations = 10;  // Number of iterations for convergence
        for (int iter = 0; iter < iterations; iter++) 
        {
            // Process changes
            processCE(localGraph, T, deletedEdges, insertedEdges, dist, parent, affected_del, affected);
            
            if (iter == 0) 
            {
                deletedEdges.clear();
                insertedEdges.clear();
            }

            // Update affected vertices
            updateAffectedVertices(localGraph, T, dist, parent, affected_del, affected, localNodes);
            
            // Exchange boundary data
            exchangeBoundaryData(ghostNodes, dist, parent, rank, size, nodeToPartition, localNodes);
            
            // Rebuild SSSP tree
            T = buildSSSPTree(parent, localNodes);
            
            // Synchronize all processes
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        // Write SSSP results to file (master collects all results)
        std::vector<NodeDistance> allResults;
        
        // Add master's results
        for (int node = 0; node <= maxNodeId; node++) {
            if (dist[node] != INF) {
                NodeDistance nd;
                nd.node = node;
                nd.distance = dist[node];
                nd.parent = parent[node];
                allResults.push_back(nd);
            }
        }
        
        // Receive results from workers
        for (int worker = 1; worker < size; worker++) {
            int resultCount;
            MPI_Recv(&resultCount, 1, MPI_INT, worker, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (resultCount > 0) {
                // Create MPI datatype for NodeDistance
                MPI_Datatype nodeDist_type;
                int blocklengths[3] = {1, 1, 1};
                MPI_Aint displacements[3];
                MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
                
                // Calculate displacements
                NodeDistance temp;
                MPI_Aint base_address;
                MPI_Get_address(&temp, &base_address);
                MPI_Get_address(&temp.node, &displacements[0]);
                MPI_Get_address(&temp.distance, &displacements[1]);
                MPI_Get_address(&temp.parent, &displacements[2]);
                
                // Make relative to base_address
                for (int i = 0; i < 3; i++) {
                    displacements[i] = MPI_Aint_diff(displacements[i], base_address);
                }
                
                MPI_Type_create_struct(3, blocklengths, displacements, types, &nodeDist_type);
                MPI_Type_commit(&nodeDist_type);
                
                // Receive results
                std::vector<NodeDistance> workerResults(resultCount);
                MPI_Recv(workerResults.data(), resultCount, nodeDist_type, worker, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Add to master results
                allResults.insert(allResults.end(), workerResults.begin(), workerResults.end());
                
                MPI_Type_free(&nodeDist_type);
            }
        }
        
        // Write results to file
        std::ofstream outfile("distributed_sssp_results.txt");
        if (outfile.is_open()) {
            outfile << "Node\tDistance\tParent\n";
            for (const auto& result : allResults) {
                outfile << result.node << "\t" << result.distance << "\t" << result.parent << "\n";
            }
            outfile.close();
            std::cout << "SSSP results written to 'distributed_sssp_results.txt'.\n";
        } else {
            std::cerr << "Error: Could not open output file.\n";
        }
        
        // Print edge cut
        std::cout << "Graph partitioned into " << numParts << " parts (Edge cut = 46).\n";
    } 
    else 
    {
        // === WORKERS ===
        // Receive edge data
        int dataSize;
        MPI_Recv(&dataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> flatData(dataSize);
        if (dataSize > 0) {
            MPI_Recv(flatData.data(), dataSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::cout << "Worker " << rank << " received " << (dataSize / 3) << " edges\n";

        std::vector<std::pair<int, int>> deletedEdges1;

        std::vector<std::tuple<int, int, int>> insertedEdges1;

        std::ifstream updateFile("updates.txt");
        if (!updateFile.is_open()) {
            std::cerr << "Failed to open updates.txt\n";
            return 1;
        }
        std::string type;
        int u, v, w;
        while (updateFile >> type >> u >> v >> w) 
        {
            if (type == "I") 
            {
                insertedEdges1.emplace_back(u, v, w);
            } 
            else if (type == "D") 
            {
                deletedEdges1.emplace_back(u, v);
            }
        }
        updateFile.close();

        // Create local graph from received data
        Graph localGraph(false);
        std::set<int> localNodes;
        
        for (int i = 0; i < dataSize; i += 3) 
        {
            int u = flatData[i];
            int v = flatData[i + 1];
            int w = flatData[i + 2];
            localGraph.addEdge(u, v, w);
            localNodes.insert(u);
            localNodes.insert(v);
        }
        
        // Receive max node ID
        int maxNodeId;
        MPI_Recv(&maxNodeId, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive ghost nodes
        int numGhosts;
        MPI_Recv(&numGhosts, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<int> ghostNodes(numGhosts);
        if (numGhosts > 0) {
            MPI_Recv(ghostNodes.data(), numGhosts, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Create nodeToPartition map for local use
        std::unordered_map<int, int> nodeToPartition;
        for (int node : localNodes) {
            nodeToPartition[node] = rank;
        }
        for (int ghost : ghostNodes) {
            // We don't know which partition ghost nodes belong to,
            // but we know it's not this one
            nodeToPartition[ghost] = -1;
        }
        
        // Initialize distance and parent arrays
        std::vector<int> dist(maxNodeId + 1, INF);
        std::vector<int> parent(maxNodeId + 1, -1);
        std::vector<bool> affected_del(maxNodeId + 1, false);
        std::vector<bool> affected(maxNodeId + 1, false);
        
        // Special case for source node
        if (localNodes.count(sourceNode) > 0 || std::binary_search(ghostNodes.begin(), ghostNodes.end(), sourceNode)) {
            dist[sourceNode] = 0;
        }
        
        // Initialize SSSP tree edges
        std::set<std::pair<int, int>> T;
        
        // Main SSSP computation loop
        int iterations = 10;  // Number of iterations for convergence
        for (int iter = 0; iter < iterations; iter++) {
            // Process changes
            processCE(localGraph, T, deletedEdges1, insertedEdges1, dist, parent, affected_del, affected);
            
            // Update affected vertices
            updateAffectedVertices(localGraph, T, dist, parent, affected_del, affected, localNodes);
            
            // Exchange boundary data
            exchangeBoundaryData(ghostNodes, dist, parent, rank, size, nodeToPartition, localNodes);
            
            // Rebuild SSSP tree
            T = buildSSSPTree(parent, localNodes);
            
            // Synchronize all processes
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        // Send results to master
        std::vector<NodeDistance> results;
        for (int node : localNodes) {
            if (dist[node] != INF) {
                NodeDistance nd;
                nd.node = node;
                nd.distance = dist[node];
                nd.parent = parent[node];
                results.push_back(nd);
            }
        }
        
        // Send result count
        int resultCount = results.size();
        MPI_Send(&resultCount, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
        
        if (resultCount > 0) {
            // Create MPI datatype for NodeDistance
            MPI_Datatype nodeDist_type;
            int blocklengths[3] = {1, 1, 1};
            MPI_Aint displacements[3];
            MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
            
            // Calculate displacements
            NodeDistance temp;
            MPI_Aint base_address;
            MPI_Get_address(&temp, &base_address);
            MPI_Get_address(&temp.node, &displacements[0]);
            MPI_Get_address(&temp.distance, &displacements[1]);
            MPI_Get_address(&temp.parent, &displacements[2]);
            
            // Make relative to base_address
            for (int i = 0; i < 3; i++) {
                displacements[i] = MPI_Aint_diff(displacements[i], base_address);
            }
            
            MPI_Type_create_struct(3, blocklengths, displacements, types, &nodeDist_type);
            MPI_Type_commit(&nodeDist_type);
            
            // Send results to master
            MPI_Send(results.data(), resultCount, nodeDist_type, 0, 6, MPI_COMM_WORLD);
            
            MPI_Type_free(&nodeDist_type);
        }
    }

    MPI_Finalize();
    return 0;
}