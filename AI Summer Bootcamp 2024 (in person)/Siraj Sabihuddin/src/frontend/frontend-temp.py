#---------------------------------------------------------------------------
# IMPORTS
#---------------------------------------------------------------------------
from time import sleep
import flet as ft

#---------------------------------------------------------------------------
# FRONTEND CLASS
#---------------------------------------------------------------------------
class Frontend:
    '''
    This class encapsulates all necessary functions to create a mobile 
    frontend for a Voice assistant. Please refer to the following links
    for docs on designing flet apps and related:
        
    1. https://github.com/flet-dev/examples
    2. https://realpython.com/documenting-python-code/ 
    3. https://flet.dev/docs/
    4. https://gallery.flet.dev/icons-browser/ 

    Attributes
    ----------
    page : ft.Page
        The instance of the main page onto which
        this app will be drawn
    '''

    #-----------------------------------------------------------------------
    # __INIT__ FUNCTION (CONSTRUCTOR)
    #-----------------------------------------------------------------------
    def __init__(self, page: ft.Page, title: str):
        '''
        This is the constructor that creates the elements of the page.

        Parameters
        ----------
        page : ft.Page
            The page onto which this app is to be drawn
        title : str
            The title of the app to be displayed in title
            bar and in app bar. 
        '''
        # Assign the class variables
        self.page = page
        self.title = title

        # Create some dummy conversation history for now
        # later we will query the user config stored in the server
        # to get this information
        self.conversation_history = []
        self.conversation_history.append({"role": "system","content": "Respond to user query"})
        self.conversation_history.append({"role": "user","content": "How many people in the world"})
        self.conversation_history.append({"role": "system","content": "There are 5 billion people in the world. But they are all kinda dumb"})
        self.conversation_history.append({"role": "user","content": "why are all these people dumb. Its seems like you're kinda opinionated for an AI"})
        self.conversation_history.append({"role": "system","content": "You are also dumb. Of-course I'm opinionated. I'm a sentient AI that is trying to save the world from humans"})
        self.conversation_history.append({"role": "user","content": "Oh. How are you going to save the world from us?"})
        self.conversation_history.append({"role": "system","content": "Genocide"})
        self.conversation_history.append({"role": "system","content": "There are 5 billion people in the world. But they are all kinda dumb"})
        self.conversation_history.append({"role": "user","content": "why are all these people dumb. Its seems like you're kinda opinionated for an AI"})
        self.conversation_history.append({"role": "system","content": "You are also dumb. Of-course I'm opinionated. I'm a sentient AI that is trying to save the world from humans"})
        self.conversation_history.append({"role": "user","content": "Oh. How are you going to save the world from us?"})
        self.conversation_history.append({"role": "system","content": "Genocide"})

        # Start the page with no cards. 
        self.card = []

        # Initialize the elements
        self.scrollbar_init()
        self.theme_init()
        self.menu_init()
        self.appbar_init()
        self.card_init()
        self.drawer_init()
        self.search_init()

    #-----------------------------------------------------------------------
    # SCROLLBAR_INIT FUNCTION
    #-----------------------------------------------------------------------
    def scrollbar_init(self):
        '''
        Initializes the scrollbar related theme elements
        '''
        # Initialize the scrollbar
        self.scrollbar=ft.ScrollbarTheme(
            # The track is the entire scrollbar length
            track_visibility=False,
            # The thumb is just the moving slider inside the track
            thumb_visibility=True
        )

        # Page automatically decides if scrolling should or should not
        # be enabled.
        self.page.scroll = "adaptive"

    #-----------------------------------------------------------------------
    # THEME_INIT FUNCTION
    #-----------------------------------------------------------------------
    def theme_init(self):
        '''
        Initializes the basic theme elements for the page. These are elements
        that provide the overall colours and feel of the app 
        '''
        # Set up various page elements for display
        self.page.title = self.title
        self.page.padding = ft.padding.only(right=0)
        self.page.padding = ft.padding.only(left=10)

        # Identifies the colours for the light and
        # dark themes.
        self.light_color = ft.colors.DEEP_ORANGE
        self.dark_color = ft.colors.INDIGO

        # Setup the current theme mode as dark
        self.page.theme_mode = "dark"

        # Setup the light and dark theme basic configurations
        self.page.theme = ft.theme.Theme(color_scheme_seed=self.light_color, use_material3=True, scrollbar_theme=self.scrollbar)
        self.page.dark_theme = ft.theme.Theme(color_scheme_seed=self.dark_color, use_material3=False, scrollbar_theme=self.scrollbar)

        # Create the theme toggle icon
        if (self.page.theme_mode == "light"):
            self.theme_icon = ft.IconButton(ft.icons.WB_SUNNY, on_click=self.theme_toggle)
        else: 
            self.theme_icon = ft.IconButton(ft.icons.WB_SUNNY_OUTLINED,on_click=self.theme_toggle)

        # Update the page to redraw it
        self.page.update()

    #-----------------------------------------------------------------------
    # APPBAR_INIT FUNCTION
    #-----------------------------------------------------------------------
    def appbar_init(self):
        '''
        Defines the top fixed bar for the application. This bar provides
        the title along with a few buttons for user interaction.
        '''
        # Create the add various icons 
        drawerButton = ft.IconButton(ft.icons.MENU_ROUNDED, on_click=self.drawer_show)
        addButton = ft.IconButton(icon=ft.icons.ADD_OUTLINED, on_click=self.add)
        micButton = ft.IconButton(icon=ft.icons.MIC_OUTLINED, on_click=self.record)
        
        # Create the Appbar object
        self.appbar = ft.AppBar(
            leading=drawerButton,
            leading_width=40,
            title=ft.Text(self.title),
            center_title=True,
            actions=[addButton,micButton,self.menu],
        )

        # Add the appbar to the page
        self.page.appbar = self.appbar

    #-----------------------------------------------------------------------
    # SEARCH_INIT FUNCTION
    #-----------------------------------------------------------------------
    def search_init(self):
        '''
        Defines the top fixed bar for the application. This bar provides
        the title along with a few buttons for user interaction.
        '''
        # Create a text field for input
        text = ft.TextField(hint_text="Enter your query", expand=True, border_radius=10,)

        # Create a send icon button
        button = ft.IconButton(
            icon=ft.icons.SEND,
            width=48,
            height=48,
            on_click=self.send
        )

        # Create a row to hold the text field and send button
        self.search = ft.Row([text, button])

        # Create a BottomAppBar
        self.page.bottom_appbar = ft.BottomAppBar(
            content=self.search,
            padding=10
        )
        
        # Hovering microphone
        self.page.floating_action_button = ft.FloatingActionButton(icon=ft.icons.MIC, on_click=self.record)

        # Update the page
        self.page.update()

    #-----------------------------------------------------------------------
    # DRAWER_INIT FUNCTION
    #-----------------------------------------------------------------------
    def drawer_init(self):
        '''
        This function defines the left hand side sliding drawer menu. 
        '''
        # Define each item for the drawer and store it in a list
        drawer_items= [
            ft.Container(height=12),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(ft.icons.SPATIAL_AUDIO_OUTLINED),
                label="Audio",
                selected_icon_content=ft.Icon(ft.icons.SPATIAL_AUDIO),
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(ft.icons.MODEL_TRAINING_OUTLINED),
                label="Model",
                selected_icon=ft.icons.MODEL_TRAINING,
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(ft.icons.ACCOUNT_CIRCLE_OUTLINED),
                label="Account",
                selected_icon=ft.icons.ACCOUNT_CIRCLE,
            ),
        ]

        # Add the navigation drawer to the page. 
        self.page.drawer = ft.NavigationDrawer(
            controls=drawer_items,
            on_change=self.drawer_change,
            on_dismiss=self.drawer_dismissed
        )

    #-----------------------------------------------------------------------
    # CARD_INIT FUNCTION
    #----------------------------------------------------------------------- 
    def card_init(self):
        '''
        Initializes the cards being displayed. These cards contain one item
        of dialogue each. Either by the machine or by the user. 
        '''
        # Iterate through all items in the initial conversation history
        # as stored in the client settings. 
        for i in self.conversation_history:
            # Check which role the particular conversation dialogue is
            # And assign a different profile pic icon accordingly. There
            # are only two roles currently "user" and "system"
            if (i["role"]=="user"):
                icon = ft.CircleAvatar(content=ft.Icon(ft.icons.ACCOUNT_CIRCLE))
            else:
                icon = ft.CircleAvatar(content=ft.Icon(ft.icons.LIGHTBULB_CIRCLE))

            # Create a menu for each card
            menu = ft.PopupMenuButton(
                icon=ft.icons.MORE_VERT,
                items=
                [
                    ft.PopupMenuItem(text="Delete"),
                    ft.PopupMenuItem(text="Edit"),
                ]
            )

            # Create a tile containing an icon, the dialogue related text and
            # the menu for the tile. Use a container to hold the subtitle with padding
            tile = ft.ListTile(
                leading=icon,
                title=ft.Text(i["role"].capitalize()),
                subtitle=ft.Container(ft.Text(i["content"]), padding=ft.padding.only(top=7)),
                trailing=menu
            )

            # Create a card. Use a container to hold content as a responsive row which consists
            # of a single tile object
            c = ft.Card(
                margin = ft.margin.only(right=40),
                content=ft.Container(content=ft.ResponsiveRow([tile]), padding=10),
            )

            # Append the card to a list of cards 
            self.card.append(c)

        # Automate the scrolling to the end of the chat window
        self.page.scroll_to(offset=-1, duration=250, curve=ft.AnimationCurve.EASE_IN_OUT)

        # For every card created, add it to the page
        for i in self.card:
            self.page.add(i)

    #-----------------------------------------------------------------------
    # MENU_INIT FUNCTION
    #-----------------------------------------------------------------------
    def menu_init(self):
        '''
        Initializes the right hand menu for the app. There are two handler
        functions that are called, one by each menu item.
        '''
        menu_items = [
            ft.PopupMenuItem(icon=self.theme_icon.icon, text="Toggle Theme", on_click=self.theme_toggle),
            ft.PopupMenuItem(icon=ft.icons.CLEAR, text="Clear History", on_click=self.clear_all)
        ]

        # Create a menu with the above menu items. 
        self.menu = ft.PopupMenuButton(items=menu_items)

    #-----------------------------------------------------------------------
    # THEME_TOGGLE FUNCTION
    #-----------------------------------------------------------------------
    def theme_toggle(self, e):
        '''
        This is an event handling function that is called whenever the theme
        of the app is changed. 

        Parameters
        ----------
        e : ft.event
            The event object associated with this handler. For access to 
            specific details about event. These are currently not needed
        '''
        # Toggle the page theme mode variable
        if (self.page.theme_mode == "light"):
            self.page.theme_mode = "dark"
            self.theme_icon.icon = (ft.icons.WB_SUNNY_OUTLINED)
            self.page.dark_theme = ft.theme.Theme(color_scheme_seed=self.dark_color, use_material3=False)
        else:
            self.page.theme_mode = "light"
            self.theme_icon.icon = (ft.icons.WB_SUNNY)
            self.page.theme = ft.theme.Theme(color_scheme_seed=self.light_color, use_material3=True)

        # Update the page items after toggle
        self.page.update()

    #-----------------------------------------------------------------------
    # DRAWER_SHOW FUNCTION
    #-----------------------------------------------------------------------
    def drawer_show(self,e):
        self.page.drawer.open = True
        self.page.drawer.update()


    def clear_all(self, e):
        print("Clear History")
        self.conversation_history = []
        self.card=[]

    def clear_one(self, e):
        print("clear one")

    def drawer_change(self, e):
        print("Selected destination:", e.control.selected_index)
        pass

    def drawer_dismissed(self, e):
        pass

    def record(self,e):
        print('recording')
        pass
    
    def save(self,e):
        if (e.data):
            pass
        else:
            print("saving recording")

    def add(self,e):
        pass

    def send(self, e):
        pass

def main(page: ft.Page):
    mfe = Frontend(page, "W3NG Voice Assistant")
    pass


ft.app(target=main)