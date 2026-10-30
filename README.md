# MyFlamma

![description](icons/MyFlamma_logo.png)

[Español](#español) | [English](#english)

## English

### Description
**MyFlamma** is an open-source QGIS plugin designed to process and classify aerial LiDAR point clouds automatically and subsequently export generated results as Flammap inputs for fire simulations and lightweight 3D OBJ models.

### Installation
1. Download the plugin by clicking on `<Code> → Download ZIP`, at the top of the repository.

2. In QGIS, select `Plugins → Manage and Install Plugins → Install from ZIP`.

3. Drag and drop the ZIP file onto the menu's sole form field.

4. Install the required dependencies (see [dependencies](#dependencies)).

5. Restart QGIS.

6. Manually activate the plugin from ``Plugins → Manage and Install Plugins → Installed``.

<!-- Manual installation -->

<!-- 1.Copy the plugin repository into your own QGIS plugin directory (in this example, the `plugins` folder):
   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

   **Note:** The paths above correspond to the **default** QGIS profile. If you are using a custom profile, replace `default` with your profile name (e.g., `profiles/my_profile/python/plugins/`).

  2. Restart QGIS.

  3. Manually activate the plugin from ``Plugins → Manage and Install Plugins``.-->

### Dependencies
This plugin requires the following Python libraries:
- **laspy** for reading and writing LAS/LAZ files.
- **GDAL** and **OSR** for geospatial data manipulation.
- **numpy** for numerical calculations and statistical queries.
- **scikit-learn** for point clustering.
- **scipy** for scientific data processing and analysis.
- **matplotlib** for generating graphs and visualisations.
- **reportlab** for generating PDF reports.

Currently, dependencies are installed automatically when importing the plugin as a ZIP file. However, in case of failure, it is possible to install them manually:

### Install dependencies

#### Windows - QGIS installed alongside OSGeo4W
  1. Open OSGeo4W Shell
  2. Paste and run the following command:
  ```bash
  python -m pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
  ```

**Note**: make sure the command is executed inside the QGIS path. It should look something like this:

```bash
C:\PROGRA~1\QGISX>
```

#### macOS
 1. Open the macOS terminal
 2. Navigate to the following path:
    ```bash
    /Applications/"Your_QGIS_Version".app/Contents/MacOS/bin/
    ```
 3. Paste and execute the following command:
    ```bash
    pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
    ```
 4. Update the sip and PyQt5-sip modules to ensure compatibility between PyQt and QGIS.
    ```bash
    pip install sip --upgrade
    ```	  
    ```bash
    pip install PyQt5-sip --upgrade
    ```

#### Linux (Ubuntu/Debian)
 1. Open the Linux terminal
 2. Paste and execute the following command:
    ```bash
    pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
    ```

### Availability
This plugin is available for QGIS versions >3.18 to 3.99 (`develop_3.0` branch) and 4.0 (`main` branch). We recommend using stable versions to get the most out of its functionality.

### Acknowledgements
Plugin developed in collaboration with the "Grupo de Ingeniería de Medios (GIM)" of the Cáceres' School of Technology.

---

## Español

### Descripción

**MyFlamma** es un complemento de código abierto para QGIS diseñado para procesar y clasificar automáticamente nubes de puntos LiDAR aéreas y, posteriormente, exportar los resultados generados como datos de entrada para Flammap, con el fin de realizar simulaciones de incendios y crear modelos 3D ligeros en formato OBJ.

### Instalación
1. Descargue el complemento haciendo clic en ``<Código> → Descargar ZIP``, en la parte superior del repositorio.

2. En QGIS, seleccione ``Complementos → Administrar e instalar complementos → Instalar desde ZIP``.

3. Arrastre y suelte el archivo ZIP en el único campo del formulario del menú.

4. Instale las dependencias necesarias (consulta [dependencias](#dependencias)).

5. Reinicie QGIS.

6. Active manualmente el complemento desde ``Complementos → Administrar e instalar complementos → Instalados``.

<!-- Instalación manual -->

<!-- 1. Copie la carpeta del complemento en su directorio de complementos QGIS propio (en este ejemplo, la carpeta `plugins`):
   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

   **Nota:** Las rutas anteriores corresponden al perfil **predeterminado** de QGIS. Si utiliza un perfil personalizado, sustituya «predeterminado» por el nombre de su perfil (por ejemplo, «perfiles/mi_perfil/python/plugins/»).

2. Reinicie QGIS.
3. Active manualmente el complemento desde **Complementos > Administrar e instalar complementos**. -->

### Dependencias
Este complemento requiere las siguientes bibliotecas de Python:
- **laspy** para leer y escribir archivos LAS/LAZ.
- **GDAL** y **OSR** para la manipulación de datos geoespaciales.
- **numpy** para cálculos numéricos y consultas estadísticas.
- **scikit-learn** para la clusterización de puntos.
- **scipy** para el procesamiento y análisis de datos científicos.
- **matplotlib** para la generación de gráficos y visualizaciones.
- **reportlab** para la generación de informes en PDF.

Actualmente, las dependencias se instalan de manera automática en la importación del plugin como ZIP. En caso de fallo, sin embargo, es posible instalarlas manualmente:

### Instalar dependencias 

#### Windows - QGIS instalado junto con OSGeo4W
  1. Abra OSGeo4W Shell.
  2. Pegue y ejecute el siguiente comando:
  ```bash
  python -m pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
  ```

**Nota**: asegúrese de que el comando se ejecuta dentro de la ruta de QGIS. Esta debería tener un aspecto similar al siguiente:

```bash
C:\PROGRA~1\QGISX>
```

#### macOS
 1. Abra el terminal de macOS
 2. Desplácese a la siguiente ruta:
    ```bash
    /Applications/«Tu_versión_de_QGIS».app/Contents/MacOS/bin/
    ```
 3. Pegue y ejecute el siguiente comando:
    ```bash
    pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
    ```
 4. Actualice los módulos sip y PyQt5-sip para asegurar la compatibilidad entre PyQt y QGIS
    ```bash
    pip install sip --upgrade
    ```	  
    ```bash
    pip install PyQt5-sip --upgrade
    ```

#### Linux (Ubuntu/Debian)
 1. Abra el terminal de Linux
 2. Pegue y ejecute el siguiente comando:
    ```bash
    pip install laspy[lazrs,laszip] scipy GDAL OSR matplotlib reportlab scikit-learn shapely
    ```

### Disponibilidad
Este plugin está disponible para versiones de QGIS >3.18 a 3.99 (rama `develop_3.0`) y 4.0 (rama `main`). Se recomienda el uso de versiones estable para aprovechar su funcionalidad al máximo.

### Agradecimientos
Complemento desarrollado en colaboración con el Grupo de Ingeniería de Medios (GIM) de la Escuela Politécnica de Cáceres.
