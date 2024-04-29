from core import create_engine
# from trame.widgets import vtk
def main(server=None, **kwargs):
    engine = create_engine(server)
    engine.server.start(**kwargs)

if __name__ == "__main__":
    main()
