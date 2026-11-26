"""Serialize portable USD assets directly from Blender's evaluated datablocks.

This writer needs no pxr package and works on Blender builds without the USD operator.
"""
import json
import re

import bpy


def number(value):
    return format(float(value), '.8g')


def vector(values):
    return '(' + ', '.join(number(v) for v in values) + ')'


def matrix(value):
    # USD uses row vectors. Blender's matrix multiplies column vectors.
    return '(' + ', '.join(vector(row) for row in value.transposed()) + ')'


def identifier(name):
    return re.sub('[^A-Za-z0-9_]', '_', name)


def quoted(value):
    return json.dumps(str(value), ensure_ascii=True)


def write_material(f, name, data):
    path=f'/Materials/{name}'
    maps=data.get('maps') or dict(zip(('Base','Normal'),[{'file':x,'channel':'rgb'} for x in data['files']]))
    names={'Base':('diffuseColor','color3f'),'Normal':('normal','normal3f'),
           'Roughness':('roughness','float'),'Metallic':('metallic','float')}
    f.write(f'''    def Material "{name}" {{
        token outputs:surface.connect = <{path}/Preview.outputs:surface>
        def Shader "Preview" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:emissiveColor = {vector(v*data['emission'] for v in data['color'])}
            token outputs:surface
''')
    for tag,(n,ty) in names.items():
        if tag in maps:
            f.write(f'            {ty} inputs:{n}.connect = <{path}/{tag}.outputs:{maps[tag]["channel"]}>\n')
        elif tag=='Base':
            f.write(f'            color3f inputs:diffuseColor = {vector(data["color"])}\n')
        elif tag in {'Roughness','Metallic'}:
            f.write(f'            float inputs:{n} = {number(data[n])}\n')
    f.write('        }\n')
    if maps:
        f.write('''        def Shader "UV" {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
''')
        for tag,m in maps.items():
            cs='sRGB' if tag=='Base' else 'raw'
            f.write(f'''        def Shader "{tag}" {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @Textures/{m['file']}@
            token inputs:sourceColorSpace = "{cs}"
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = <{path}/UV.outputs:result>
            float3 outputs:rgb
            float outputs:r
            float outputs:g
            float outputs:b
''')
            if tag=='Normal':
                f.write('            float4 inputs:scale = (2, 2, 2, 1)\n')
                f.write('            float4 inputs:bias = (-1, -1, -1, 0)\n')
            f.write('        }\n')
    f.write('    }\n')


def write_mesh(f, ob, graph, prototype):
    evaluated = ob.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        f.write(f'        def Mesh "{identifier(ob.name)}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{\n')
        f.write('            uniform token subdivisionScheme = "none"\n')
        f.write('            point3f[] points = ['+', '.join(vector(v.co) for v in mesh.vertices)+']\n')
        f.write('            int[] faceVertexCounts = ['+', '.join(str(p.loop_total) for p in mesh.polygons)+']\n')
        f.write('            int[] faceVertexIndices = ['+', '.join(str(l.vertex_index) for l in mesh.loops)+']\n')
        if mesh.polygons:
            f.write('            normal3f[] normals = ['+', '.join(vector(n.vector) for n in mesh.corner_normals)+'] (interpolation = "faceVarying")\n')
        if mesh.uv_layers.active:
            f.write('            texCoord2f[] primvars:st = ['+', '.join(vector(u.vector) for u in mesh.uv_layers.active.uv)+'] (interpolation = "faceVarying")\n')
        if evaluated.material_slots and evaluated.material_slots[0].material:
            mat_name=evaluated.material_slots[0].material.name
            f.write(f'            rel material:binding = </Prototypes/{prototype}/Looks/{mat_name}>\n')
        if len(evaluated.material_slots)>1:
            f.write('            token subsetFamily:materialBind:familyType = "nonOverlapping"\n')
            for index,slot in enumerate(evaluated.material_slots):
                polygons=[str(p.index) for p in mesh.polygons if p.material_index==index]
                if not polygons or not slot.material:
                    continue
                f.write(f'            def GeomSubset "Material_{index}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{\n')
                f.write('                uniform token elementType = "face"\n')
                f.write('                uniform token familyName = "materialBind"\n')
                f.write('                int[] indices = ['+', '.join(polygons)+']\n')
                f.write(f'                rel material:binding = </Prototypes/{prototype}/Looks/{slot.material.name}>\n')
                f.write('            }\n')
        f.write(f'            matrix4d xformOp:transform = {matrix(ob.matrix_local)}\n')
        f.write('            uniform token[] xformOpOrder = ["xformOp:transform"]\n')
        f.write('        }\n')
    finally:
        evaluated.to_mesh_clear()


def write_camera(f, scene, ob, keys=None):
    d=ob.data
    factor=.01   # Convert mm to tenths of stage units (1 m).
    sw=d.sensor_width*factor
    sh=sw*scene.render.resolution_y/scene.render.resolution_x*(scene.render.pixel_aspect_y/scene.render.pixel_aspect_x)
    f.write(f'''        def Camera "{ob.name}" {{
            token projection = "perspective"
            float focalLength = {number(d.lens*factor)}
            float horizontalAperture = {number(sw)}
            float verticalAperture = {number(sh)}
            float horizontalApertureOffset = {number(sw*d.shift_x)}
            float verticalApertureOffset = {number(sw*d.shift_y)}
            float2 clippingRange = {vector((d.clip_start,d.clip_end))}
''')
    if keys:
        f.write('            matrix4d xformOp:transform.timeSamples = {\n')
        for time,p,q in keys:
            transform=q.to_matrix().to_4x4()
            transform.translation=p
            f.write(f'                {number(time)}: {matrix(transform)},\n')
        f.write('            }\n')
    else:
        f.write(f'            matrix4d xformOp:transform = {matrix(ob.matrix_world)}\n')
    f.write('            uniform token[] xformOpOrder = ["xformOp:transform"]\n        }\n')


def write_usd(out, scene, assets, instances, mats, cam, twin_cam, keys, lights):
    with (out/'Materials.usda').open('w') as f:
        f.write('#usda 1.0\n\ndef Scope "Materials" {\n')
        for name,data in mats.items():
            write_material(f,name,data)
        f.write('}\n')

    with (out/'AssetPrototypes.usda').open('w') as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "Prototypes"\n    metersPerUnit = 1\n    upAxis = "Z"\n    subLayers = [@Materials.usda@]\n)\n\ndef Scope "Prototypes" {\n')
        for key,(collection,cat,description) in assets.items():
            scene.collection.children.link(collection)
            try:
                bpy.context.view_layer.update()
                graph=bpy.context.evaluated_depsgraph_get()
                f.write(f'    def Xform "{key}" {{\n')
                f.write(f'        custom string assetId = "{key}"\n')
                f.write(f'        custom string assetCategory = "{cat}"\n')
                f.write(f'        custom string assetDescription = {quoted(description)}\n')
                used={slot.material.name for ob in collection.objects if ob.type == 'MESH'
                      for slot in ob.material_slots if slot.material is not None}
                f.write('        def Scope "Looks" {\n')
                for mat in sorted(used):
                    f.write(f'            def Material "{mat}" (prepend references = @Materials.usda@</Materials/{mat}>) {{}}\n')
                f.write('        }\n')
                for ob in collection.objects:
                    if ob.type=='MESH':
                        write_mesh(f,ob,graph,key)
                f.write('    }\n')
            finally:
                scene.collection.children.unlink(collection)
        f.write('}\n')
    bpy.context.view_layer.update()

    with (out/'IndoorTwin.usda').open('w') as f:
        f.write(f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = {scene.frame_start}
    endTimeCode = {scene.frame_end}
    framesPerSecond = {scene.render.fps}
    timeCodesPerSecond = {scene.render.fps}
    subLayers = [@Materials.usda@]
)

def Xform "World" {{
''')
        for cat in dict.fromkeys(o['asset_category'] for o in instances):
            f.write(f'    def Scope "{cat}" {{\n')
            for ob in instances:
                if ob['asset_category'] != cat:
                    continue
                asset_id=ob['asset_id']
                f.write(f'''        def Xform "{identifier(ob.name)}" (
            prepend references = @AssetPrototypes.usda@</Prototypes/{asset_id}>
            instanceable = true
        ) {{
            custom string instanceId = {quoted(ob.name)}
            custom string placementEvidence = {quoted(ob['placement_evidence'])}
            matrix4d xformOp:transform = {matrix(ob.matrix_world)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
''')
            if cat=='Lighting':
                for ob in lights:
                    if ob.type!='LIGHT':
                        continue
                    d=ob.data
                    # USD intensity uses a renderer-specific photometric convention.
                    f.write(f'''        def RectLight "{identifier(ob.name)}" {{
            color3f inputs:color = {vector(d.color)}
            float inputs:intensity = {number(d.energy/(d.size*d.size_y)*.06)}
            bool inputs:normalize = false
            float inputs:width = {number(d.size)}
            float inputs:height = {number(d.size_y)}
            matrix4d xformOp:transform = {matrix(ob.matrix_world)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
''')
            f.write('    }\n')
        f.write('    def Scope "Cameras" {\n')
        write_camera(f,scene,cam,keys)
        write_camera(f,scene,twin_cam)
        f.write('    }\n}\n')
