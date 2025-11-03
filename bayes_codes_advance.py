import sys
import random
import itertools
import numpy as np
import cv2 as cv
import os

# Asegura que el script encuentre la imagen
script_dir = os.path.dirname(os.path.abspath(__file__)) 
MAP_FILE = os.path.join(script_dir, 'cape_python.png')

# Coordenadas de las áreas de búsqueda (SA)
SA1_CORNERS = (130, 265, 180, 315)  # (UL-X, UL-Y, LR-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305)
SA3_CORNERS = (105, 205, 155, 255)


class Search():
    """Juego Avanzado de Búsqueda y Rescate Bayesiano con objetivo móvil."""

    def __init__(self, name):
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print('No se pudo cargar el archivo de mapa {}'.format(MAP_FILE),
                  file=sys.stderr)
            sys.exit(1)

        # Ubicación real (secreta) del marinero
        self.area_actual = 0
        self.sailor_actual = [0, 0]

        # Arrays de NumPy para cada área
        self.sa1 = self.img[SA1_CORNERS[1]: SA1_CORNERS[3],
                            SA1_CORNERS[0]: SA1_CORNERS[2]]
        self.sa2 = self.img[SA2_CORNERS[1]: SA2_CORNERS[3],
                            SA2_CORNERS[0]: SA2_CORNERS[2]]
        self.sa3 = self.img[SA3_CORNERS[1]: SA3_CORNERS[3],
                            SA3_CORNERS[0]: SA3_CORNERS[2]]

        # Probabilidades Iniciales Aleatorias
        probs = [random.random() for _ in range(3)]
        total_prob = sum(probs)
        self.p1 = probs[0] / total_prob
        self.p2 = probs[1] / total_prob
        self.p3 = probs[2] / total_prob

        # Probabilidades de efectividad de búsqueda (SEP)
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

        # Dificultad por Área
        self.sep1_range = (0.2, 0.4)  # Área 1: Difícil
        self.sep2_range = (0.7, 0.9)  # Área 2: Fácil
        self.sep3_range = (0.4, 0.6)  # Área 3: Media

        # Matriz de Deriva de Markov
        self.drift_matrix = np.array([
            [0.8, 0.2, 0.0],  # Desde Área 1: 80% se queda, 20% va a 2
            [0.1, 0.7, 0.2],  # Desde Área 2: 10% va a 1, 70% se queda, 20% va a 3
            [0.0, 0.1, 0.9]   # Desde Área 3: 10% va a 2, 90% se queda
        ])
        # Normalización
        row_sums = self.drift_matrix.sum(axis=1)
        self.drift_matrix = self.drift_matrix / row_sums[:, np.newaxis]


    def draw_map(self, last_known):
        """Muestra el mapa base, escala, áreas y última pos. conocida."""
        # Dibuja la escala
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.putText(self.img, '50 Nautical Miles', (71, 370),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        # Dibuja y numera las áreas
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]),
                     (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '1',
                   (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]),
                     (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2',
                   (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]),
                     (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3',
                   (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)

        # Muestra la última posición conocida
        cv.putText(self.img, '+', (last_known),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = Last Known Position', (274, 355),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '* = Actual Position', (275, 370),
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 10)
        cv.waitKey(500)

    def sailor_initial_location(self):
        """Genera la ubicación INICIAL del marinero basado en las probs iniciales."""
        self.area_actual = np.random.choice([1, 2, 3], p=[self.p1, self.p2, self.p3])

        if self.area_actual == 1: shape = self.sa1.shape
        elif self.area_actual == 2: shape = self.sa2.shape
        elif self.area_actual == 3: shape = self.sa3.shape
            
        self.sailor_actual[0] = np.random.choice(shape[1]) # x local
        self.sailor_actual[1] = np.random.choice(shape[0]) # y local

    def update_sailor_location(self):
        """Actualiza la posición FÍSICA del marinero basada en la deriva."""
        current_area_index = self.area_actual - 1
        probs = self.drift_matrix[current_area_index]
        new_area = np.random.choice([1, 2, 3], p=probs)
        
        if new_area != self.area_actual:
            self.area_actual = new_area
            
            if self.area_actual == 1: shape = self.sa1.shape
            elif self.area_actual == 2: shape = self.sa2.shape
            elif self.area_actual == 3: shape = self.sa3.shape
            
            self.sailor_actual[0] = np.random.choice(shape[1])
            self.sailor_actual[1] = np.random.choice(shape[0])
            print(f"[Info: El marinero ha derivado al Área {self.area_actual}]")

    def apply_drift(self):
        """Actualiza las PROBABILIDADES (p1, p2, p3) usando la matriz de deriva."""
        p_actuales = np.array([self.p1, self.p2, self.p3])
        p_nuevas = p_actuales.dot(self.drift_matrix)
        self.p1, self.p2, self.p3 = p_nuevas
        
    def get_sailor_global_coords(self):
        """Convierte la ubicación local actual del marinero a coords. del mapa global."""
        local_x, local_y = self.sailor_actual[0], self.sailor_actual[1]
        
        if self.area_actual == 1:
            x = local_x + SA1_CORNERS[0]
            y = local_y + SA1_CORNERS[1]
        elif self.area_actual == 2:
            x = local_x + SA2_CORNERS[0]
            y = local_y + SA2_CORNERS[1]
        elif self.area_actual == 3:
            x = local_x + SA3_CORNERS[0]
            y = local_y + SA3_CORNERS[1]
        else:
            return -1, -1 # Error
            
        return x, y

    def calc_search_effectiveness(self):
        """Establece la efectividad de la búsqueda (SEP) para el turno."""
        self.sep1 = random.uniform(self.sep1_range[0], self.sep1_range[1])
        self.sep2 = random.uniform(self.sep2_range[0], self.sep2_range[1])
        self.sep3 = random.uniform(self.sep3_range[0], self.sep3_range[1])

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        """Realiza la búsqueda y devuelve los resultados."""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(itertools.product(local_x_range, local_y_range))
        random.shuffle(coords)
        
        num_searched = int(len(coords) * effectiveness_prob)
        coords = coords[:num_searched]
        
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        
        if area_num == self.area_actual and loc_actual in coords:
            return '¡Encontrado!', coords
        
        return 'No Encontrado', coords

    def revise_target_probs(self):
        """Actualiza las probabilidades (Bayes) después de una búsqueda fallida."""
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) \
                + self.p3 * (1 - self.sep3)
        
        if denom == 0: denom = 1e-10 
            
        self.p1 = self.p1 * (1 - self.sep1) / denom
        self.p2 = self.p2 * (1 - self.sep2) / denom
        self.p3 = self.p3 * (1 - self.sep3) / denom
        
        # Re-normalizar
        total = self.p1 + self.p2 + self.p3
        self.p1 /= total
        self.p2 /= total
        self.p3 /= total


def draw_menu(search_num):
    """Imprime el menú de opciones."""
    print('\nBúsqueda {}'.format(search_num))
    print(
        """
        Elige las siguientes áreas a buscar:

        0 - Salir del Juego
        1 - Buscar Área 1 dos veces
        2 - Buscar Área 2 dos veces
        3 - Buscar Área 3 dos veces
        4 - Buscar Áreas 1 y 2
        5 - Buscar Áreas 1 y 3
        6 - Buscar Áreas 2 y 3
        7 - Reiniciar Juego (ahora)
        """
        )

# --- LÓGICA PRINCIPAL DEL JUEGO ---
def main():
    # 1. Configuración inicial
    app = Search('Cape_Python')
    app.draw_map(last_known=(160, 290))
    app.sailor_initial_location() 
    
    turnos_restantes = 10 
    search_num = 1
    
    print("-" * 65)
    print("\n¡Alerta de Búsqueda y Rescate!")
    print(f"Tienes {turnos_restantes} turnos para encontrar al marinero.")
    print("\nProbabilidades (P) Iniciales de Objetivo:")
    print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))

    # 2. Bucle de turnos
    while True:
        # --- CONDICIÓN DE DERROTA ---
        if turnos_restantes == 0:
            print("\n--- ¡FIN DEL JUEGO! ---")
            print("Te has quedado sin combustible/tiempo.")
            
            sailor_x, sailor_y = app.get_sailor_global_coords()
            print(f"El marinero estaba en el Área {app.area_actual} en ({sailor_x}, {sailor_y})")
            cv.circle(app.img, (sailor_x, sailor_y), 5, (0, 0, 255), -1) 
            cv.putText(app.img, 'AQUI ESTABA', (sailor_x + 10, sailor_y + 5),
                       cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
            cv.imshow('Search Area', app.img)
            cv.waitKey(0) # Espera a que el usuario vea el resultado
            cv.destroyAllWindows() # Cierra la ventana
            return # --- CAMBIADO --- Sale de main() para que el bucle exterior reinicie

        print("-" * 65)
        print(f"\n--- Turnos Restantes: {turnos_restantes} ---")

        # 3. Fase de Deriva (antes de buscar)
        print("\nCalculando deriva del océano...")
        app.apply_drift() # Mueve las probabilidades
        app.update_sailor_location() # Mueve al marinero
        print("Nuevas probabilidades por deriva (antes de buscar):")
        print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))

        # 4. Fase de Búsqueda (elección del jugador)
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input("Elección: ")

        if choice == "0":
            print("\nSaliendo del juego...")
            cv.destroyAllWindows()
            sys.exit() # Sale completamente del programa

        elif choice == "1":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            total_pix = app.sa1.shape[0] * app.sa1.shape[1]
            app.sep1 = (len(set(coords_1 + coords_2))) / total_pix
            app.sep2 = 0
            app.sep3 = 0

        elif choice == "2":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            total_pix = app.sa2.shape[0] * app.sa2.shape[1]
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2))) / total_pix
            app.sep3 = 0

        elif choice == "3":
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            total_pix = app.sa3.shape[0] * app.sa3.shape[1]
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2))) / total_pix

        elif choice == "4":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0

        elif choice == "5":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep2 = 0

        elif choice == "6":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0

        elif choice == "7":
            print("\nReiniciando partida...")
            cv.destroyAllWindows()
            return # --- CAMBIADO --- Sale de main() para que el bucle exterior reinicie

        else:
            print("\nOpción no válida.", file=sys.stderr)
            continue
        
        # 5. Fase de Actualización (Bayes)
        if results_1 == 'No Encontrado' and results_2 == 'No Encontrado':
            app.revise_target_probs()
        
        # 6. Mostrar Resultados del Turno
        print("\nResultados de Búsqueda {}:".format(search_num))
        print("Resultado 1 = {}".format(results_1), file=sys.stderr)
        print("Resultado 2 = {}".format(results_2), file=sys.stderr)
        
        print("\nEfectividad de Búsqueda (SEP) para este turno:")
        print("SEP1 = {:.3f}, SEP2 = {:.3f}, SEP3 = {:.3f}"
              .format(app.sep1, app.sep2, app.sep3))

        # --- CONDICIÓN DE VICTORIA ---
        if results_1 == '¡Encontrado!' or results_2 == '¡Encontrado!':
            print("\n--- ¡ÉXITO! ---")
            print(f"¡Marinero encontrado en el Área {app.area_actual}!")
            sailor_x, sailor_y = app.get_sailor_global_coords()
            
            cv.circle(app.img, (sailor_x, sailor_y), 5, (255, 0, 0), -1)
            cv.putText(app.img, 'ENCONTRADO!', (sailor_x + 10, sailor_y + 5),
                       cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)
            cv.imshow('Search Area', app.img)
            cv.waitKey(0) # Espera a que el usuario vea el resultado
            cv.destroyAllWindows() # Cierra la ventana
            return # --- CAMBIADO --- Sale de main() para que el bucle exterior reinicie
        
        # 7. Continuar al siguiente turno
        else:
            print("\nNuevas Probabilidades (P) para la Búsqueda {}:"
                  .format(search_num + 1))
            print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}"
                  .format(app.p1, app.p2, app.p3))

        turnos_restantes -= 1
        search_num += 1
    
# --- BUCLE DE JUEGO PRINCIPAL (MODIFICADO) ---
if __name__ == '__main__':
    # Este bucle 'while True' se asegura de que main() se llame 
    # de nuevo cada vez que termina (a menos que se use sys.exit()).
    while True:
        main()
        print("\n" + "=" * 65)
        print("           CARGANDO NUEVA PARTIDA...           ")
        print("=" * 65 + "\n")