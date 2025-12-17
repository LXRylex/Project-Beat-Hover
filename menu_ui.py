from ursina import *
import random, math
import beat_hover

app = Ursina()

window.title = "Beat Hover"
window.color = color.black
camera.position = (0, 0, -15)
camera.look_at((0, 0, 0))


# ----------------- helpers -----------------
def rgba(r, g, b, a=255):
    return color.rgba(r/255, g/255, b/255, a/255)

def pulse(t, speed=2.0):
    return (math.sin(t * speed) + 1) * 0.5


# ----------------- safe futuristic button (NO scale on hover) -----------------
class NeonButton(Button):
    def __init__(self, text="", **kwargs):
        if "scale" not in kwargs:
            kwargs["scale"] = (0.62, 0.10)

        super().__init__(
            text=text,
            model="quad",
            color=rgba(20, 20, 26, 210),
            highlight_color=rgba(20, 20, 26, 210),
            pressed_color=rgba(16, 16, 22, 235),
            text_color=color.white,
            text_origin=(0, 0),
            collider="box",
            **kwargs
        )

        self._shine = Entity(
            parent=self, model="quad",
            scale=(0.98, 0.60),
            position=(0, 0.10, -0.01),
            color=rgba(255, 255, 255, 10),
        )

        self._glow = Entity(
            parent=self, model="quad",
            scale=(1.06, 1.35),
            z=0.02,
            color=rgba(155, 135, 255, 0),
        )

        self._border = Entity(
            parent=self, model="quad",
            scale=(1.01, 1.10),
            z=0.01,
            color=rgba(255, 255, 255, 18),
        )

        self._base_y = self.y
        self._hovering = False
        self.on_mouse_enter = self._enter
        self.on_mouse_exit  = self._exit

    def _enter(self):
        if self._hovering:
            return
        self._hovering = True
        self.animate_y(self._base_y + 0.006, duration=0.10, curve=curve.out_quad)
        self._glow.animate_color(rgba(155, 135, 255, 85), duration=0.10, curve=curve.out_quad)
        self._border.animate_color(rgba(255, 255, 255, 40), duration=0.10, curve=curve.out_quad)
        self._shine.animate_color(rgba(255, 255, 255, 18), duration=0.10, curve=curve.out_quad)

    def _exit(self):
        self._hovering = False
        self.animate_y(self._base_y, duration=0.12, curve=curve.out_quad)
        self._glow.animate_color(rgba(155, 135, 255, 0), duration=0.14, curve=curve.out_quad)
        self._border.animate_color(rgba(255, 255, 255, 18), duration=0.14, curve=curve.out_quad)
        self._shine.animate_color(rgba(255, 255, 255, 10), duration=0.14, curve=curve.out_quad)


# ----------------- MENU ROOT (THIS is what we enable/disable) -----------------
menu_root = Entity(parent=camera.ui)
menu_root.enabled = True


# ----------------- background: starfield (parented to menu_root) -----------------
bg_parent = Entity(parent=menu_root)

# IMPORTANT: UI background must be DISABLED when game starts, otherwise it covers the world.
bg = Entity(parent=bg_parent, model="quad", scale=(2, 2), z=1, color=rgba(0, 0, 0, 255))

stars = []
STAR_COUNT = 90
for _ in range(STAR_COUNT):
    s = Entity(
        parent=bg_parent,
        model="quad",
        color=rgba(255, 255, 255, random.randint(25, 110)),
        scale=random.uniform(0.0035, 0.010),
        position=(random.uniform(-1.1, 1.1), random.uniform(-1.1, 1.1), 0.9),
    )
    s._spd = random.uniform(0.06, 0.20)
    s._drift = random.uniform(-0.02, 0.02)
    stars.append(s)

vignette = Entity(parent=bg_parent, model="quad", scale=(2, 2), z=0.8, color=rgba(0, 0, 0, 130))


# ----------------- menu card (also parented to menu_root) -----------------
menu_parent = Entity(parent=menu_root)
menu_parent.enabled = True

card_shadow = Entity(parent=menu_parent, model="quad", scale=(0.92, 0.72), position=(0, -0.01, 0.30), color=rgba(0, 0, 0, 140))
card = Entity(parent=menu_parent, model="quad", scale=(0.90, 0.70), position=(0, 0, 0.25), color=rgba(14, 14, 18, 210))
card_border = Entity(parent=menu_parent, model="quad", scale=(0.905, 0.705), position=(0, 0, 0.24), color=rgba(255, 255, 255, 18))

title_glow = Text(text="BEAT HOVER", parent=menu_parent, origin=(0, 0), y=0.20, scale=2.4, color=rgba(160, 140, 255, 90))
title = Text(text="", parent=menu_parent, origin=(0, 0), y=0.20, scale=2.15, color=color.white)

subtitle = Text(
    text="Project of boredom",
    parent=menu_parent,
    origin=(0, 0),
    y=0.12,
    scale=0.9,
    color=rgba(190, 190, 200, 170),
)

play_button = NeonButton(parent=menu_parent, text="PLAY (RANDOM)", position=(0, 0.02, 0.20))
load_button = NeonButton(parent=menu_parent, text="LOAD SONG (WAV/MP3)", position=(0, -0.12, 0.20))
quit_button = NeonButton(parent=menu_parent, text="QUIT", position=(0, -0.26, 0.20), scale=(0.42, 0.10))

hint = Text(
    text="ESC in-game: back to menu   |   SPACE in-game: mouse lock",
    parent=menu_parent,
    origin=(0, 0),
    y=-0.36,
    scale=0.75,
    color=rgba(170, 170, 180, 140),
)

# entrance animation
menu_parent.scale = 0.965
menu_parent.y = -0.02
invoke(lambda: menu_parent.animate_scale(1.0, duration=0.22, curve=curve.out_back), delay=0.02)
invoke(lambda: menu_parent.animate_y(0.0, duration=0.22, curve=curve.out_quad), delay=0.02)


# ----------------- show/hide menu properly -----------------
def show_menu():
    menu_root.enabled = True
    mouse.visible = True

    menu_parent.scale = 0.98
    menu_parent.y = -0.01
    menu_parent.animate_scale(1.0, duration=0.18, curve=curve.out_quad)
    menu_parent.animate_y(0.0, duration=0.18, curve=curve.out_quad)

def hide_menu_then(action_fn):
    # animate the card a bit, then fully disable ALL menu visuals
    menu_parent.animate_scale(0.985, duration=0.10, curve=curve.in_out_quad)
    menu_parent.animate_y(-0.01, duration=0.10, curve=curve.in_out_quad)
    invoke(lambda: _really_hide_and_go(action_fn), delay=0.10)

def _really_hide_and_go(action_fn):
    menu_root.enabled = False   # <--- critical fix (hides bg + stars + card)
    mouse.visible = False
    action_fn()


# ----------------- init game -----------------
beat_hover.init_game(on_exit_to_menu=show_menu)
show_menu()


# ----------------- buttons -----------------
play_button.on_click = lambda: hide_menu_then(beat_hover.start_random)
load_button.on_click = lambda: hide_menu_then(beat_hover.load_song_dialog)
quit_button.on_click = application.quit


# ----------------- update/input -----------------
def update():
    dt = time.dt

    # only animate stars while menu is visible
    if menu_root.enabled:
        for s in stars:
            s.y -= s._spd * dt
            s.x += s._drift * dt
            if s.y < -1.15:
                s.y = 1.15
                s.x = random.uniform(-1.1, 1.1)
                s._spd = random.uniform(0.06, 0.20)
                s._drift = random.uniform(-0.02, 0.02)

        t = time.time()
        p = pulse(t, 1.8)
        title.scale = 2.15 + p * 0.05
        title_glow.scale = 2.4 + p * 0.07
        a = int(70 + p * 60)
        title_glow.color = rgba(160, 140, 255, a)
        menu_parent.y = math.sin(t * 0.9) * 0.006

    # forward game update always
    beat_hover.game_update()

def input(key):
    if key == "escape" and menu_root.enabled:
        application.quit()
        return
    beat_hover.game_input(key)


app.run()
