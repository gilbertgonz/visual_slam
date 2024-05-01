import OpenGL.GL as gl
import pangolin
import numpy as np

def draw_trajectory(trajectory):
    gl.glLineWidth(2)
    gl.glColor3f(0.0, 0.0, 1.0)  # Blue color for the trajectory
    pangolin.DrawLines(trajectory)

def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    # Generate sample trajectory
    trajectory = np.array([[0, -6, 6]])
    for i in range(300):
        trajectory = np.vstack((trajectory, trajectory[-1] + np.random.random(3) - 0.5))

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        # Draw the trajectory
        draw_trajectory(trajectory)

        # Render OpenGL Cube
        pangolin.glDrawColouredCube(0.1)

        pangolin.FinishFrame()

if __name__ == '__main__':
    main()
