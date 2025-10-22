import numpy as np
import matplotlib.pyplot as plt
import time

class GravityBody:
    def __init__(self, mass: float, position: (float, float, float), velocity: (float, float, float)):
        self.mass = mass
        self.position = position
        self.velocity = velocity

class GravitySimulation:
    C_G = 1000.0
    C_Y0 = -1.70241438392
    C_Y1 = 1.35120719196
    C_Y2 = 0.67560359598
    C_Y3 = -0.17560359598

    def __init__(self, bodies: list[GravityBody], do_orbital_elems: bool = False):
        self.count = len(bodies)
        self.do_orbital_elems = do_orbital_elems

        self.vector_temp = np.empty((self.count, 3), dtype=np.float32)
        self.scalar_temp = np.empty((self.count, 1), dtype=np.float32)

        self.masses = np.empty_like(self.scalar_temp)
        self.positions = np.empty_like(self.vector_temp)
        self.velocities = np.empty_like(self.vector_temp)
        self.forces = np.empty_like(self.vector_temp)
        self.accels = np.empty_like(self.vector_temp)

        self.idiag = np.diag_indices(self.count, 2)
        self.iforces = np.empty((self.count, self.count, 3), dtype=np.float32)
        self.idisps = np.empty((self.count, self.count, 3), dtype=np.float32)
        self.idists = np.empty((self.count, self.count), dtype=np.float32)

        for i, body in enumerate(bodies):
            self.masses[i] = body.mass
            self.positions[i] = body.position
            self.velocities[i] = body.velocity

        self.cog_mass = np.zeros(1, dtype=np.float32)
        self.cog_position = np.zeros(3, dtype=np.float32)
        self.cog_velocity = np.zeros(3, dtype=np.float32)


        
        self.w_dirs = np.empty_like(self.vector_temp)
        self.p_dirs = np.empty_like(self.vector_temp)
        self.q_dirs = np.empty_like(self.vector_temp)
        self.angms = np.empty_like(self.scalar_temp)
        self.eccs = np.empty_like(self.scalar_temp)



    # Take sum of all forces by all bodies 
    def calc_forces_accels(self):
        # Compute displacements
        np.copyto(self.idisps, np.reshape(self.positions, (1, self.count, 3)))
        self.idisps -= np.reshape(self.positions, (self.count, 1, 3))

        # Copy displacements to forces
        np.copyto(self.iforces, self.idisps)
        
        # Take self dot product of displacements and get distance cubed
        self.idisps *= self.idisps
        np.sum(self.idisps, axis=2, out=self.idists)
        self.idists **= 1.5

        # Calculate normalized displacement / distance squared
        with np.errstate(divide='ignore', invalid='ignore'): # Ignore divide by zero
            self.iforces /= np.reshape(self.idists, (self.count, self.count, 1))

        # Set diagonal to zero to remove NaNs 
        # self.iforces[self.idiag] = (0.0, 0.0, 0.0)

        # Remove NaNs
        np.nan_to_num(self.iforces, copy=False, posinf=0.0, neginf=0.0)

        # Multuply by body masses and gravitational constant
        self.iforces *= np.reshape(self.masses, (1, self.count, 1))
        self.iforces *= np.reshape(self.masses, (self.count, 1, 1))
        self.iforces *= GravitySimulation.C_G

        # Sum individual forces into total force
        np.sum(self.iforces, axis=1, out=self.forces)

        # Divide force by mass to get acceleration
        np.divide(self.forces, self.masses, out=self.accels) # self.accels = self.forces / self.masses

    def calc_cog_state(self):
        # Calculate COG mass (sum of body masses)
        np.sum(self.masses, axis=0, out=self.cog_mass)

        # Calculate COG position (weighted mean of body positions)
        np.copyto(self.vector_temp, self.positions)
        self.vector_temp *= self.masses
        np.sum(self.vector_temp, axis=0, out=self.cog_position)
        self.cog_position /= self.cog_mass

        # Calculate COG velocity (weighted mean of body velocities)
        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= self.masses
        np.sum(self.vector_temp, axis=0, out=self.cog_velocity)
        self.cog_velocity /= self.cog_mass

    # Use the 4th order Yoshida integrator to integrate positions and velocities while conserving angular momentum
    def leapfrog_integrate(self, dt: float):
        if dt == 0.0: return

        # Step 1a: r1 = r0 + c2*v0*dt
        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= dt * GravitySimulation.C_Y2
        self.positions += self.vector_temp

        # Step 1b: v1 = v0 + c1*a(r1)*dt
        self.calc_forces_accels()
        np.copyto(self.vector_temp, self.accels)
        self.vector_temp *= dt * GravitySimulation.C_Y1
        self.velocities += self.vector_temp

        # Step 2a: r2 = r1 + d3*v1*dt
        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= dt * GravitySimulation.C_Y3
        self.positions += self.vector_temp

        # Step 2b: v2 = v1 + c0*a(r2)*dt
        self.calc_forces_accels()
        np.copyto(self.vector_temp, self.accels)
        self.vector_temp *= dt * GravitySimulation.C_Y0
        self.velocities += self.vector_temp

        # Step 3a: r3 = r2 + d3*v2*dt
        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= dt * GravitySimulation.C_Y3
        self.positions += self.vector_temp

        # Step 3b: v3 = v2 + c1*a(r3)*dt
        self.calc_forces_accels()
        np.copyto(self.vector_temp, self.accels)
        self.vector_temp *= dt * GravitySimulation.C_Y1
        self.velocities += self.vector_temp

        # Step 4a: r4 = r3 + c2*v3*dt
        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= dt * GravitySimulation.C_Y2
        self.positions += self.vector_temp

    def euler_integrate(self, dt: float):
        self.calc_forces_accels()

        np.copyto(self.vector_temp, self.velocities)
        self.vector_temp *= dt
        self.positions += self.vector_temp

        np.copyto(self.vector_temp, self.accels)
        self.vector_temp *= dt
        self.velocities += self.vector_temp

        self.vector_temp *= dt * 0.5
        self.positions += self.vector_temp

    def update_orbital_elems(self):
        if not self.do_orbital_elems: return

        mu = self.cog_mass * GravitySimulation.C_G
        rel_positions = self.positions - self.cog_position
        rel_velocities = self.velocities - self.cog_velocity

        self.w_dirs = np.cross(rel_positions, rel_velocities, axis=1)
        self.p_dirs = np.cross(rel_velocities, self.w_dirs, axis=1)

        self.angms = np.reshape(np.linalg.norm(self.w_dirs, axis=1), (self.count, 1))
        self.w_dirs /= self.angms

        self.p_dirs /= mu
        self.p_dirs -= rel_positions / np.reshape(np.linalg.norm(rel_positions, axis=1), (self.count, 1))
        self.eccs = np.reshape(np.linalg.norm(self.p_dirs, axis=1), (self.count, 1))
        self.p_dirs /= self.eccs

        self.q_dirs = np.cross(self.w_dirs, self.p_dirs)



    def step(self, dt: float, max_dt: float = 0.02):
        # Integrate over timestep to get final positions and velocities

        while dt > max_dt:
            self.leapfrog_integrate(max_dt)
            dt -= max_dt
        self.leapfrog_integrate(dt)
        # self.euler_integrate(dt)

        # Calculate total force and acceleration at final position
        self.calc_forces_accels()

        # print(self.forces, end="\n\n")

        # Calculate COG state
        self.calc_cog_state()

        self.update_orbital_elems()

    def body_mass(self, i: int):
        return self.masses[i, 0]

    def body_position(self, i: int, relative: bool = False):
        if relative:
            return self.positions[i, :] - self.cog_position
        else:
            return self.positions[i, :]

    def body_velocity(self, i: int, relative: bool = False):
        if relative:
            return self.velocities[i, :] - self.cog_velocity
        else:
            return self.velocities[i, :]

    def body_accel(self, i: int):
        return self.accels[i, :]

    def body_force(self, i: int):
        return self.forces[i, :]

    def generate_orbit_plot(self, i: int, v: np.ndarray) -> np.ndarray:
        cos, sin = np.cos(v), np.sin(v)
        orbit_parameter = self.orbit_parameter(i)
        distance = orbit_parameter / (1 + self.eccs[i, :] * cos)
        return np.reshape(distance, (len(v), 1)) * (
            np.reshape(self.p_dirs[i, :], (1, 3)) * np.reshape(cos, (len(v), 1)) 
            + 
            np.reshape(self.q_dirs[i, :], (1, 3)) * np.reshape(sin, (len(v), 1))
        )

    def orbit_parameter(self, i: int):
        return (self.angms[i, :]**2) / (GravitySimulation.C_G * self.cog_mass)

    def orbit_periapsis(self, i: int):
        return (self.orbit_parameter(i) / (1 + self.eccs[i, :])) * self.p_dirs[i, :]
    
    def orbit_apoapsis(self, i: int):
        return (self.orbit_parameter(i) / (1 - self.eccs[i, :])) * -self.p_dirs[i, :]

    def orbit_period(self, i: int):

        return (self.angms[i, 0]**3 * 2) / ((GravitySimulation.C_G * self.cog_mass)**2 * (1 - self.eccs[i, 0]**2))




def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        import math
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        # cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        # cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        # cur_zrange = (cur_zlim[1] - cur_zlim[0])*.5
        # xdata = event.xdata # get event x location
        # ydata = event.ydata # get event y location
        # zdata = event.zdata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([cur_xlim[0]*scale_factor,
                     cur_xlim[1]*scale_factor])
        ax.set_ylim([cur_ylim[0]*scale_factor,
                     cur_ylim[1]*scale_factor])
        ax.set_zlim([cur_zlim[0]*scale_factor,
                     cur_zlim[1]*scale_factor])
        plt.draw() # force re-draw
        
    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def do_graph(sim: GravitySimulation):

    fig = plt.figure()
    axes = fig.add_subplot(projection="3d")

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)

    axes.set_xlim(-400, 400)
    axes.set_ylim(-400, 400)
    axes.set_zlim(-400, 400)
    plt.axis("off")
    plt.grid(visible=None)

    zoom_factory(axes, base_scale=1.15)

    should_stop = [False]

    def on_close(event):
        should_stop[0] = True

    fig.canvas.mpl_connect('close_event', on_close)

    plt.ioff()
    plt.show(block=False)

    total_time = 0.0

    v = np.linspace(-np.pi, np.pi, num=40, dtype=np.float32)

    points = []
    lines = []
    tstart = time.time()
    while not should_stop[0]:
        for point in points:
            point.remove()
        points.clear()

        for line in lines:
            line.remove()
        lines.clear()

        dt = max(time.time()-tstart, 0.0)
        total_time += dt
        sim.step(dt)

        # print(sim.w_dirs)
        # print(sim.p_dirs)
        # print(sim.q_dirs)
        
        for i in range(sim.count):
            mass = sim.body_mass(i)
            (px, py, pz) = sim.body_position(i, True)
            (vx, vy, vz) = sim.body_velocity(i, True)
            (ax, ay, az) = sim.body_accel(i)
            (fx, fy, fz) = sim.body_force(i)

            if i != 0:
                orbit = sim.generate_orbit_plot(i, v)
                periapsis = sim.orbit_periapsis(i)
                apoapsis = sim.orbit_apoapsis(i)
                # print(orbit)
                print(sim.orbit_period(i))
                lines.extend(axes.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color='black'))
                points.append(axes.scatter(periapsis[0], periapsis[1], periapsis[2], color='green'))
                points.append(axes.scatter(apoapsis[0], apoapsis[1], apoapsis[2], color='red'))


            lines.extend(axes.plot([px, px+vx], [py, py+vy], [pz, pz+vz], color='blue'))
            lines.extend(axes.plot([px, px+ax], [py, py+ay], [pz, pz+az], color='green'))
            # lines.extend(axes.plot([px, px+fx], [py, py+fy], [pz, pz+fz], color='red'))
            points.append(axes.scatter([px], [py], [pz], color='gray'))

        print()

        fig.canvas.draw()

        tstart = time.time()
        
        plt.pause(1/60)

# sim = GravitySimulation([
#     GravityBody(10000.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
#     GravityBody(100.0, (198.3, 0.0, 26.11), (0.0, 230.0, 0.0)),
#     GravityBody(90.0, (448.3, 0.0, -39.22), (0.0, 150.0, 0.0)),
#     GravityBody(80.0, (700.0, 0.0, 0.0), (0.0, 120.0, 0.0)),
# ], do_orbital_elems=True)

sim = GravitySimulation([
    GravityBody(10000.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    GravityBody(100.0, (200.0, 0.0, 0.0), (0.0, 230.0, 0.0)),
    GravityBody(90.0, (220.0, 0.0, 0.0), (0.0, 150.0, 0.0)),
    GravityBody(80.0, (700.0, 0.0, 0.0), (0.0, 120.0, 0.0)),
], do_orbital_elems=True)

# sim.step(0.0)

# print("Step 0:")
# print(sim.positions)
# print(sim.velocities)
# print(sim.forces)
# print(sim.accels)

# start = time.time()
# sim.step(0.001)
# print(f"v--- took {time.time()-start} seconds")

# print("Step 1:")
# print(sim.positions)
# print(sim.velocities)
# print(sim.forces)
# print(sim.accels)

# start = time.time()
# sim.step(0.001)
# print(f"v--- took {time.time()-start} seconds")

# print("Step 2:")
# print(sim.positions)
# print(sim.velocities)
# print(sim.forces)
# print(sim.accels)

do_graph(sim)