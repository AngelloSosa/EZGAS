from flask import Flask, render_template, request, jsonify, redirect, url_for
import networkx as nx
import numpy as np
import requests
import os
from geopy.distance import geodesic
import csv
from networkx.algorithms.components import connected_components
from heapq import heappop, heappush
import math

app = Flask(__name__)
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Diccionario para almacenar los subgrafos de cada localidad
subgrafos = {}

datos_y_subgrafos_cargados = False

@app.route('/load_province', methods=['POST'])
def load_province():
    provincia = request.json.get('provincia', '').title()  # Provincia solicitada
    print(f"Cargando datos para la provincia: {provincia}")  # Log para depuración

    # Leer el CSV solo para la provincia solicitada
    estaciones = []
    with open('app/spanish_gas_stations.csv', mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('provincia', '').title() == provincia:
                estaciones.append({
                    "latitud": float(row['latitud']),
                    "longitud": float(row['longitud']),
                    "municipio": row['municipio'],
                    "rotulo": row['rotulo']
                })

    # Si no hay datos para la provincia
    if not estaciones:
        print(f"No se encontraron estaciones para la provincia {provincia}")
        return jsonify({"error": f"No se encontraron datos para la provincia {provincia}"}), 404

    # Crear grafo con las estaciones
    grafo = crear_grafo(estaciones)
    subgrafos[provincia] = grafo  # Almacenar en el diccionario

    print(f"Grafo para la provincia {provincia} cargado con {len(grafo.nodes)} nodos")
    return jsonify({"message": f"Grafo de {provincia} cargado correctamente", "nodes": len(grafo.nodes)})

    
def crear_grafo(estaciones):
    """Crea un grafo con las estaciones de servicio de una provincia."""
    G = nx.Graph()

    # Agregar nodos (una estación por nodo)
    for i, estacion in enumerate(estaciones):
        G.add_node(i, **estacion)

    # Conectar nodos con un peso de 1 por defecto (sin distancias geográficas)
    for i in range(len(estaciones)):
        for j in range(i + 1, len(estaciones)):
            G.add_edge(i, j, weight=1)  # Conectar todos los nodos con peso 1

    return G

@app.route('/get_subgraph', methods=['POST'])
def get_subgraph():
    """
    Obtiene el subgrafo correspondiente a la provincia ingresada.
    """
    provincia = request.json.get('provincia', '').title()  # Convertir a título por consistencia
    verificar_datos_y_subgrafos_cargados()

    if provincia in subgrafos:
        # Extraer el subgrafo de la provincia
        subgrafo = subgrafos[provincia]

        # Obtener nodos del subgrafo para enviarlos como respuesta
        estaciones = [
            {
                "id": node,
                "latitud": data['latitud'],
                "longitud": data['longitud'],
                "rotulo": data['rotulo']
            }
            for node, data in subgrafo.nodes(data=True)
        ]

        return jsonify({
            "provincia": provincia,
            "estaciones": estaciones,
            "mensaje": "Subgrafo identificado correctamente."
        })
    else:
        return jsonify({"error": "Provincia no encontrada."}), 404
    
def obtener_datos_trafico(lat, lng):
    """Obtiene datos de tráfico vehicular usando Google Maps Traffic API."""
    API_KEY = api_key
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json"
    
    # Configura una solicitud a la API de tráfico
    params = {
        "lat": lat,
        "lng": lng,
        "radius": 10000,  # Radio en metros para obtener datos
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Retorna algún factor de tráfico: velocidad promedio, congestión, etc.
        # Aquí asumimos que la API devuelve un indicador de tráfico, ajustar según la API usada.
        return data.get("traffic_factor", 1.0)  # Ejemplo: retorna 1.0 si no hay tráfico
    return 1.0  # Por defecto, sin penalización por tráfico

def heuristic(node1, node2, traffic_factor=1.0):
    """Heurística ajustada con tráfico vehicular."""
    distance = geodesic(
        (node1['latitud'], node1['longitud']), 
        (node2['latitud'], node2['longitud'])
    ).kilometers
    # Ajustamos la heurística con un factor de tráfico
    return distance * traffic_factor

def a_star(graph, start, goal):
    """Implementación del algoritmo A* con heurística dinámica."""
    open_set = []
    heappush(open_set, (0, start))  # (coste, nodo)
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(graph.nodes[start], graph.nodes[goal])

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            # Reconstruir el camino
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Camino de inicio a fin

        for neighbor in graph.neighbors(current):
            # Obtener datos de tráfico dinámicamente para el nodo vecino
            traffic_factor = obtener_datos_trafico(
                graph.nodes[neighbor]['latitud'], 
                graph.nodes[neighbor]['longitud']
            )
            tentative_g_score = g_score[current] + graph.edges[current, neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(graph.nodes[neighbor], graph.nodes[goal], traffic_factor)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No se encontró un camino

def prim_mst(graph, start):
    """Implementación del algoritmo Prim para calcular la ruta óptima."""
    mst = nx.Graph()
    visited = set()
    edges = [(0, start, None)]  # (peso, nodo actual, nodo anterior)

    while edges:
        weight, current, prev = heappop(edges)
        if current in visited:
            continue

        visited.add(current)
        if prev is not None:
            mst.add_edge(prev, current, weight=weight)

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                heappush(edges, (graph.edges[current, neighbor]['weight'], neighbor, current))

    return mst

@app.route('/find_closest_station', methods=['POST'])
def find_closest_station():
    """
    Encuentra la estación más cercana en el subgrafo usando A*.
    """
    data = request.json
    provincia = data.get('provincia', '').title()
    user_lat = float(data['lat'])
    user_lng = float(data['lng'])

    verificar_datos_y_subgrafos_cargados()

    if provincia not in subgrafos:
        return jsonify({"error": "Provincia no encontrada."}), 404

    subgrafo = subgrafos[provincia]

    # Nodo temporal para la ubicación del usuario
    user_node = len(subgrafo.nodes)
    subgrafo.add_node(user_node, latitud=user_lat, longitud=user_lng)

    # Aplicar A* para encontrar la estación más cercana
    closest_station = None
    closest_path = None
    min_distance = float('inf')

    for target in subgrafo.nodes:
        if target == user_node:
            continue
        path = a_star(subgrafo, user_node, target)
        if path:
            distance = sum(subgrafo.edges[path[i], path[i + 1]]['weight'] for i in range(len(path) - 1))
            if distance < min_distance:
                min_distance = distance
                closest_station = target
                closest_path = path

    # Eliminar nodo temporal
    subgrafo.remove_node(user_node)

    if closest_station is not None:
        station_data = subgrafo.nodes[closest_station]
        return jsonify({
            "closest_station": {
                "latitud": station_data['latitud'],
                "longitud": station_data['longitud'],
                "rotulo": station_data.get('rotulo', '')
            },
            "path": closest_path,
            "distance": min_distance
        })
    else:
        return jsonify({"error": "No se encontró una estación cercana."})

@app.route('/calculate_prim_route', methods=['POST'])
def calculate_prim_route():
    """
    Calcula la ruta óptima usando Prim dentro del subgrafo.
    """
    data = request.json
    provincia = data.get('provincia', '').title()
    user_lat = float(data['lat'])
    user_lng = float(data['lng'])
    target_lat = float(data['target_lat'])
    target_lng = float(data['target_lng'])

    verificar_datos_y_subgrafos_cargados()

    if provincia not in subgrafos:
        return jsonify({"error": "Provincia no encontrada."}), 404

    subgrafo = subgrafos[provincia]

    # Añadir nodos temporales (usuario y objetivo)
    user_node = len(subgrafo.nodes)
    target_node = user_node + 1
    subgrafo.add_node(user_node, latitud=user_lat, longitud=user_lng)
    subgrafo.add_node(target_node, latitud=target_lat, longitud=target_lng)

    # Calcular MST con Prim
    mst = prim_mst(subgrafo, user_node)

    # Eliminar nodos temporales
    subgrafo.remove_node(user_node)
    subgrafo.remove_node(target_node)

    # Convertir el MST a un formato JSON para el cliente
    mst_edges = [
        {
            "from": edge[0],
            "to": edge[1],
            "weight": data['weight']
        }
        for edge, data in mst.edges(data=True)
    ]

    return jsonify({
        "mst": mst_edges,
        "mensaje": "Ruta óptima calculada correctamente."
    })

def verificar_datos_y_subgrafos_cargados(provincia=None):
    """
    Verifica si los datos y subgrafos están cargados.
    Si se especifica una provincia, verifica solo esa provincia.
    """
    global datos_y_subgrafos_cargados

    if provincia:
        # Verificar si el subgrafo de la provincia ya está cargado
        if provincia.title() in subgrafos:
            return True
        else:
            return False
    else:
        # Verificar si al menos un subgrafo está cargado
        if datos_y_subgrafos_cargados and len(subgrafos) > 0:
            return True
        else:
            return False


@app.route('/')
def landing():
    """Página de aterrizaje."""
    return render_template('landing.html')

@app.route('/index')
def index():
    """Página principal con la lista de provincias."""
    verificar_datos_y_subgrafos_cargados()
    provincias = list(subgrafos.keys())
    ordenar = request.args.get('ordenar')
    if ordenar == 'alfabeticamente':
        provincias.sort()
    return render_template('index.html', provincias=provincias)

@app.route('/grafo/<provincia>')
def mostrar_grafo(provincia):
    """Muestra un mapa interactivo con las estaciones de la provincia."""
    verificar_datos_y_subgrafos_cargados()
    provincia = provincia.title()  # Convertir a título
    if provincia not in subgrafos:
        return f"No hay datos para la provincia: {provincia}", 404

    # Extraer los datos de las estaciones en la provincia
    estaciones = [
        {"latitud": data['latitud'], "longitud": data['longitud']}
        for _, data in subgrafos[provincia].nodes(data=True)
    ]

    return render_template('graph.html', provincia=provincia, estaciones=estaciones)

@app.route('/filtered_gas_stations', methods=['GET'])
def filtered_gas_stations():
    provincia = request.args.get('provincia', '').title()
    user_lat = float(request.args.get('lat'))
    user_lng = float(request.args.get('lng'))
    rotulo = request.args.get('rotulo', '').strip().upper()

    if provincia not in subgrafos:
        return jsonify({"error": "Provincia no encontrada."}), 404

    subgrafo = subgrafos[provincia]
    search_radius = 50
    filtered_stations = []

    for _, data in subgrafo.nodes(data=True):
        if rotulo and data.get('rotulo', '').strip().upper() != rotulo:
            continue

        distance = geodesic((user_lat, user_lng), (data['latitud'], data['longitud'])).kilometers
        if distance <= search_radius:
            filtered_stations.append({
                "latitud": data['latitud'],
                "longitud": data['longitud'],
                "rotulo": data['rotulo'],
                "direccion": data.get('direccion', 'Sin dirección')
            })

    return jsonify(filtered_stations)

@app.route('/buscar', methods=['POST'])
def buscar():
    """Ruta para buscar una provincia específica."""
    provincia = request.form.get('provincia').title()  # Convertir a título
    if provincia in subgrafos:
        return jsonify({"url": f"/grafo/{provincia}"})
    return jsonify({"error": "Localidad no encontrada"}), 404

@app.route('/ruta')
def mostrar_ruta():
    """Muestra un mapa interactivo para ingresar la ubicación del usuario y ver la ruta."""
    # Extraer los datos de todas las estaciones
    estaciones = []
    for subgrafo in subgrafos.values():
        estaciones.extend([
            {"latitud": data['latitud'], "longitud": data['longitud']}
            for _, data in subgrafo.nodes(data=True)
        ])
    return render_template('route.html', estaciones=estaciones, api_key=os.getenv("GOOGLE_MAPS_API_KEY"))
