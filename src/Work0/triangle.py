import taichi as ti
import math

ti.init(arch=ti.cpu, debug=True)

NUM_VERTICES = 3
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    model_matrix = ti.Matrix([
        [c,  -s,  0.0, 0.0],
        [s,   c,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return model_matrix

@ti.func
def get_view_matrix(eye_pos: ti.types.vector(3)):
    view_matrix = ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return view_matrix

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    persp_to_ortho = ti.Matrix([
        [n, 0.0, 0.0,    0.0],
        [0.0, n, 0.0,    0.0],
        [0.0, 0.0, n+f, -n*f],
        [0.0, 0.0, 1.0,    0.0]
    ])
    
    ortho_scale = ti.Matrix([
        [2.0/(r-l), 0.0,     0.0,     0.0],
        [0.0,     2.0/(t-b), 0.0,     0.0],
        [0.0,     0.0,     2.0/(n-f), 0.0],
        [0.0,     0.0,     0.0,     1.0]
    ])
    
    ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2.0],
        [0.0, 1.0, 0.0, -(t+b)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    ortho_matrix = ortho_scale @ ortho_trans
    proj_matrix = ortho_matrix @ persp_to_ortho
    return proj_matrix

@ti.kernel
def compute_screen_coords(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    fov = 45.0
    aspect_ratio = 1.0
    zNear = 0.1
    zFar = 50.0
    
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(fov, aspect_ratio, zNear, zFar)
    
    mvp_matrix = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v3 = vertices[i]
        v4 = ti.Vector([v3[0], v3[1], v3[2], 1.0])
        v_clip = mvp_matrix @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    window_size = (700, 700)
    gui = ti.GUI("3D MVP Transformation", res=window_size, background_color=0xFFFFFF)
    
    rotation_angle = 0.0
    
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                rotation_angle += 10.0
            elif e.key == 'd':
                rotation_angle -= 10.0
        
        compute_screen_coords(rotation_angle)
        
        p0 = screen_coords[0]
        p1 = screen_coords[1]
        p2 = screen_coords[2]
        
        gui.line(p0, p1, radius=2, color=0xFF0000)
        gui.line(p1, p2, radius=2, color=0x00FF00)
        gui.line(p2, p0, radius=2, color=0x0000FF)
        
        gui.show()

if __name__ == "__main__":
    main()