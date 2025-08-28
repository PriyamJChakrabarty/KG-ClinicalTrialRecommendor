from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import json

URI = "neo4j://your-server-address:7687"
AUTH = ("your-username", "your-password")

def project_graph(tx):
    query = """
    CALL gds.graph.project(
        'clinicalTrialsGraph',
        ['SubjectNode', 'ObjectNode'],
        {
            RELATIONSHIP: {orientation: 'UNDIRECTED'}
        }
    )
    """
    tx.run(query)

def is_graph_projected(tx):
    query = """
    CALL gds.graph.exists('clinicalTrialsGraph')
    YIELD exists
    RETURN exists
    """
    result = tx.run(query)
    return result.single()["exists"]

def reproject_graph(tx):
    query = """
    CALL gds.graph.drop('clinicalTrialsGraph', false)
    """
    tx.run(query)
    project_graph(tx)

def run_louvain_clustering(tx):
    query = """
    CALL gds.louvain.write('clinicalTrialsGraph', {
        writeProperty: 'community',
        includeIntermediateCommunities: false
    })
    YIELD communityCount, modularity
    RETURN communityCount, modularity
    """
    result = tx.run(query)
    record = result.single()
    return record["communityCount"], record["modularity"]

def get_community_members(tx, community_id):
    query = """
    MATCH (n:SubjectNode)
    WHERE n.community = $community_id
    RETURN n.name AS trial_id
    """
    result = tx.run(query, community_id=community_id)
    return [record["trial_id"] for record in result]

def get_node_community(tx, trial_id):
    query = """
    MATCH (n:SubjectNode {name: $trial_id})
    RETURN n.community AS community
    """
    result = tx.run(query, trial_id=trial_id)
    record = result.single()
    return record["community"] if record else None

def create_similarity_relationship(tx, trial1, trial2, similarity):
    query = """
    MATCH (n1:SubjectNode {name: $trial1}), (n2:SubjectNode {name: $trial2})
    MERGE (n1)-[r:SIMILAR_TO]->(n2)
    SET r.similarity = $similarity
    MERGE (n2)-[r2:SIMILAR_TO]->(n1)
    SET r2.similarity = $similarity
    """
    tx.run(query, trial1=trial1, trial2=trial2, similarity=similarity)

def get_cached_similarity(tx, trial1, trial2):
    query = """
    MATCH (n1:SubjectNode {name: $trial1})-[r:SIMILAR_TO]->(n2:SubjectNode {name: $trial2})
    RETURN r.similarity AS similarity
    """
    result = tx.run(query, trial1=trial1, trial2=trial2)
    record = result.single()
    return record["similarity"] if record else None

def get_node_neighbors(tx, trial_id):
    query = """
    MATCH (n:SubjectNode {name: $trial_id})-[:RELATIONSHIP]-(obj:ObjectNode)
    RETURN obj.name AS neighbor
    """
    result = tx.run(query, trial_id=trial_id)
    return set(record["neighbor"] for record in result)

def get_intermediate_similarities(tx, trial_id, community_members):
    query = """
    MATCH (n1:SubjectNode {name: $trial_id})-[r:SIMILAR_TO]->(n2:SubjectNode)
    WHERE n2.name IN $community_members
    RETURN n2.name AS trial, r.similarity AS similarity
    """
    result = tx.run(query, trial_id=trial_id, community_members=community_members)
    return {record["trial"]: record["similarity"] for record in result}

def compute_optimized_jaccard(tx, trial1, trial2, community_members, intermediate_sims):
    cached_sim = get_cached_similarity(tx, trial1, trial2)
    if cached_sim is not None:
        return cached_sim
    
    neighbors1 = get_node_neighbors(tx, trial1)
    neighbors2 = get_node_neighbors(tx, trial2)
    
    intersection = neighbors1 & neighbors2
    union = neighbors1 | neighbors2
    
    if len(union) == 0:
        similarity = 0.0
    else:
        base_jaccard = len(intersection) / len(union)
        
        boost_factor = 0.0
        for intermediate_trial in community_members:
            if intermediate_trial in [trial1, trial2]:
                continue
            
            if intermediate_trial in intermediate_sims:
                sim_to_intermediate = intermediate_sims[intermediate_trial]
            else:
                sim_to_intermediate = get_cached_similarity(tx, trial1, intermediate_trial)
                if sim_to_intermediate is None:
                    continue
            
            other_sim = get_cached_similarity(tx, trial2, intermediate_trial)
            if other_sim is not None:
                boost_factor += min(sim_to_intermediate, other_sim) * 0.1
        
        similarity = min(1.0, base_jaccard + boost_factor)
    
    create_similarity_relationship(tx, trial1, trial2, similarity)
    return similarity

def find_similar_trials_optimized(tx, trial_id, top_k=10):
    community_id = get_node_community(tx, trial_id)
    if community_id is None:
        return []
    
    community_members = get_community_members(tx, community_id)
    if trial_id in community_members:
        community_members.remove(trial_id)
    
    intermediate_sims = get_intermediate_similarities(tx, trial_id, community_members)
    
    similarities = []
    
    for other_trial in community_members:
        similarity = compute_optimized_jaccard(tx, trial_id, other_trial, community_members, intermediate_sims)
        similarities.append({"trial": other_trial, "similarity": similarity})
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]

def check_node_exists(tx, node_name, label):
    query = f"""
    MATCH (n:{label} {{name: TRIM($node_name)}})
    RETURN COUNT(n) > 0 AS exists
    """
    result = tx.run(query, node_name=node_name.strip())
    exists = result.single()["exists"]
    return exists

def main():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        with driver.session() as session:
            graph_exists = session.read_transaction(is_graph_projected)

            if graph_exists:
                session.write_transaction(reproject_graph)
            else:
                session.write_transaction(project_graph)

            print("Running Louvain clustering...")
            community_count, modularity = session.write_transaction(run_louvain_clustering)
            print(f"Found {community_count} communities with modularity {modularity:.4f}")

            trial_id = input("Enter the trial ID (e.g., NCT00752622): ")

            node_exists = session.read_transaction(check_node_exists, trial_id, 'SubjectNode')
            if not node_exists:
                print(f"Node '{trial_id}' does not exist in the graph. Please enter a valid trial ID.")
                return

            print(f"Finding trials similar to {trial_id}...")
            similar_trials = session.read_transaction(find_similar_trials_optimized, trial_id)

            if similar_trials:
                print("Top 10 similar trials:")
                for i, trial in enumerate(similar_trials, 1):
                    print(f"{i}. Trial ID: {trial['trial']}, Similarity: {trial['similarity']:.4f}")
            else:
                print("No similar trials found.")
    except ServiceUnavailable as e:
        print(f"Failed to connect to Neo4j: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()