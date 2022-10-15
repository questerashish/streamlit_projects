import streamlit as st
from datetime import time as dtm, datetime
import pandas as pd
import numpy as np
import time

# st.set_page_config(layout="wide")
# st.title('st.progress')

# with st.expander('About this app'):
#      st.write('You can now display the progress of your calculations in a Streamlit app with the `st.progress` command.')

# my_bar = st.progress(0)

# for percent_complete in range(100):
#      time.sleep(0.05)
#      my_bar.progress(percent_complete + 1)

# st.balloons()


# st.header('st.slider')

# # Example 1

# st.subheader('Slider')

# age = st.slider('How old are you?', 0, 130, 25)
# st.write("I'm ", age, 'years old')

# # Example 2

# st.subheader('Range slider')

# values = st.slider(
#      'Select a range of values',
#      0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)

# # Example 3

# st.subheader('Range time slider')

# appointment = st.slider(
#      "Schedule your appointment:",
#      value=(dtm(11, 30), dtm(12, 45)))
# st.write("You're scheduled for:", appointment)

# # Example 4

# st.subheader('Datetime slider')

# start_time = st.slider(
#      "When do you start?",
#      value=datetime(2020, 1, 1, 9, 30),
#      format="MM/DD/YY - hh:mm")
# st.write("Start time:", start_time)


# st.header('Line chart')

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)


# import streamlit as st

# st.header('st.selectbox')

# option = st.selectbox(
#      'What is your favorite color?',
#      ('Blue', 'Red', 'Green'))

# st.write('Your favorite color is ', option)

# import streamlit as st

# st.header('st.multiselect')

# options = st.multiselect(
#      'What are your favorite colors',
#      ['Green', 'Yellow', 'Red', 'Blue'],
#      ['Yellow', 'Red'])

# st.write('You selected:', options)


# import streamlit as st
# import pandas as pd
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# st.header('`streamlit_pandas_profiling`')

# df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

# @st.cache(suppress_st_warning=True)
# def read_rep():
#     pr = df.profile_report()
#     st_profile_report(pr)
# read_rep()

# import streamlit as st

# st.header('st.latex')

# st.latex(r'''
#      a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#      \sum_{k=0}^{n-1} ar^k =
#      a \left(\frac{1-r^{n}}{1-r}\right)
#      ''')

# import streamlit as st
# import pandas as pd

# st.title('st.file_uploader')

# st.subheader('Input CSV')
# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   st.subheader('DataFrame')
#   st.write(df)
#   st.subheader('Descriptive Statistics')
#   st.write(df.describe())
# else:
#   st.info('☝️ Upload a CSV file')






# st.title('How to layout your Streamlit app')

# with st.expander('About this app'):
#   st.write('This app shows the various ways on how you can layout your Streamlit app.')
#   st.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=250)

# st.sidebar.header('Input')
# user_name = st.sidebar.text_input('What is your name?')
# user_emoji = st.sidebar.selectbox('Choose an emoji', ['', '😄', '😆', '😊', '😍', '😴', '😕', '😱'])
# user_food = st.sidebar.selectbox('What is your favorite food?', ['', 'Tom Yum Kung', 'Burrito', 'Lasagna', 'Hamburger', 'Pizza'])

# st.header('Output')

# col1, col2, col3 = st.columns(3)

# with col1:
#   if user_name != '':
#     st.write(f'👋 Hello {user_name}!')
#   else:
#     st.write('👈  Please enter your **name**!')

# with col2:
#   if user_emoji != '':
#     st.write(f'{user_emoji} is your favorite **emoji**!')
#   else:
#     st.write('👈 Please choose an **emoji**!')

# with col3:
#   if user_food != '':
#     st.write(f'🍴 **{user_food}** is your favorite **food**!')
#   else:
#     st.write('👈 Please choose your favorite **food**!')


# import streamlit as st

# st.title('st.form')

# # Full example of using the with notation
# st.header('1. Example of using `with` notation')
# st.subheader('Coffee machine')

# with st.form('my_form'):
#     st.subheader('**Order your coffee**')

#     # Input widgets
#     coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
#     coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
#     brewing_val = st.selectbox('Brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
#     serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
#     milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
#     owncup_val = st.checkbox('Bring own cup')

#     # Every form must have a submit button
#     submitted = st.form_submit_button('Submit')

# if submitted:
#     st.markdown(f'''
#         ☕ You have ordered:
#         - Coffee bean: `{coffee_bean_val}`
#         - Coffee roast: `{coffee_roast_val}`
#         - Brewing: `{brewing_val}`
#         - Serving type: `{serving_type_val}`
#         - Milk: `{milk_val}`
#         - Bring own cup: `{owncup_val}`
#         ''')
# else:
#     st.write('☝️ Place your order!')


# # Short example of using an object notation
# st.header('2. Example of object notation')

# form = st.form('my_form_2')
# selected_val = form.slider('Select a value')
# form.form_submit_button('Submit')

# st.write('Selected value: ', selected_val)

# import streamlit as st

# st.title('st.experimental_get_query_params')

# with st.expander('About this app'):
#   st.write("`st.experimental_get_query_params` allows the retrieval of query parameters directly from the URL of the user's browser.")

# # 1. Instructions
# st.header('1. Instructions')
# st.markdown('''
# In the above URL bar of your internet browser, append the following:
# `?name=Jack&surname=Beanstalk`
# after the base URL `http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/`
# such that it becomes 
# `http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/?firstname=Jack&surname=Beanstalk`
# ''')


# # 2. Contents of st.experimental_get_query_params
# st.header('2. Contents of st.experimental_get_query_params')
# st.write(st.experimental_get_query_params())


# # 3. Retrieving and displaying information from the URL
# st.header('3. Retrieving and displaying information from the URL')

# firstname = st.experimental_get_query_params()['firstname'][0]
# surname = st.experimental_get_query_params()['surname'][0]

# st.write(f'Hello **{firstname} {surname}**, how are you?')



# import streamlit as st
# import numpy as np
# import pandas as pd
# from time import time

# st.title('st.cache')

# # Using cache
# a0 = time()
# st.subheader('Using st.cache')

# @st.cache(suppress_st_warning=True)
# def load_data_a():
#   df = pd.DataFrame(
#     np.random.rand(2000000, 5),
#     columns=['a', 'b', 'c', 'd', 'e']
#   )
#   return df

# st.write(load_data_a())
# a1 = time()
# st.info(a1-a0)


# # Not using cache
# b0 = time()
# st.subheader('Not using st.cache')

# def load_data_b():
#   df = pd.DataFrame(
#     np.random.rand(2000000, 5),
#     columns=['a', 'b', 'c', 'd', 'e']
#   )
#   return df

# st.write(load_data_b())
# b1 = time()
# st.info(b1-b0)


# import streamlit as st

# st.title('st.session_state')

# def lbs_to_kg():
#   st.session_state.kg = st.session_state.lbs/2.2046
# def kg_to_lbs():
#   st.session_state.lbs = st.session_state.kg*2.2046

# st.header('Input')
# col1, spacer, col2 = st.columns([2,1,2])
# with col1:
#   pounds = st.number_input("Pounds:", key = "lbs", on_change = lbs_to_kg)
# with col2:
#   kilogram = st.number_input("Kilograms:", key = "kg", on_change = kg_to_lbs)

# st.header('Output')
# st.write("st.session_state object:", st.session_state)



# import streamlit as st
# import requests

# st.title('🏀 Bored API app')

# st.sidebar.header('Input')
# selected_type = st.sidebar.selectbox('Select an activity type', ["education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"])
# selected_partc = st.sidebar.selectbox('Select no of participants', ["1","2","3","4","5"])
# # selected_type = st.sidebar.selectbox('Select an activity type', ["education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"])



# suggested_activity_url = f'http://www.boredapi.com/api/activity?type={selected_type}?participants={selected_partc}'
# json_data = requests.get(suggested_activity_url)
# suggested_activity = json_data.json()

# c1, c2 = st.columns(2)
# with c1:
#   with st.expander('About this app'):
#     st.write('Are you bored? The **Bored API app** provides suggestions on activities that you can do when you are bored. This app is powered by the Bored API.')
# with c2:
#   with st.expander('JSON data'):
#     st.write(suggested_activity)

# st.header('Suggested activity')
# st.info(suggested_activity['activity'])

# col1, col2, col3 = st.columns(3)
# with col1:
#   st.metric(label='Number of Participants', value=suggested_activity['participants'], delta='')
# with col2:
#   st.metric(label='Type of Activity', value=suggested_activity['type'].capitalize(), delta='')
# with col3:
#   st.metric(label='Price', value=suggested_activity['price'], delta='')



# First, we will need the following imports for our application.

# import json
# import streamlit as st
# from pathlib import Path

# # As for Streamlit Elements, we will need all these objects.
# # All available objects and there usage are listed there: https://github.com/okld/streamlit-elements#getting-started

# from streamlit_elements import elements, dashboard, mui, editor, media, lazy, sync, nivo

# # Change page layout to make the dashboard take the whole page.

# st.set_page_config(layout="wide")

# with st.sidebar:
#     st.title("🗓️ #30DaysOfStreamlit")
#     st.header("Day 27 - Streamlit Elements")
#     st.write("Build a draggable and resizable dashboard with Streamlit Elements.")
#     st.write("---")

#     # Define URL for media player.
#     media_url = st.text_input("Media URL", value="https://www.youtube.com/watch?v=vIQQR_yq-8I")

# # Initialize default data for code editor and chart.
# #
# # For this tutorial, we will need data for a Nivo Bump chart.
# # You can get random data there, in tab 'data': https://nivo.rocks/bump/
# #
# # As you will see below, this session state item will be updated when our
# # code editor change, and it will be read by Nivo Bump chart to draw the data.

# if "data" not in st.session_state:
#     st.session_state.data = Path("data.json").read_text()

# # Define a default dashboard layout.
# # Dashboard grid has 12 columns by default.
# #
# # For more information on available parameters:
# # https://github.com/react-grid-layout/react-grid-layout#grid-item-props

# layout = [
#     # Editor item is positioned in coordinates x=0 and y=0, and takes 6/12 columns and has a height of 3.
#     dashboard.Item("editor", 0, 0, 6, 3),
#     # Chart item is positioned in coordinates x=6 and y=0, and takes 6/12 columns and has a height of 3.
#     dashboard.Item("chart", 6, 0, 6, 3),
#     # Media item is positioned in coordinates x=0 and y=3, and takes 6/12 columns and has a height of 4.
#     dashboard.Item("media", 0, 2, 12, 4),
# ]

# # Create a frame to display elements.

# with elements("demo"):

#     # Create a new dashboard with the layout specified above.
#     #
#     # draggableHandle is a CSS query selector to define the draggable part of each dashboard item.
#     # Here, elements with a 'draggable' class name will be draggable.
#     #
#     # For more information on available parameters for dashboard grid:
#     # https://github.com/react-grid-layout/react-grid-layout#grid-layout-props
#     # https://github.com/react-grid-layout/react-grid-layout#responsive-grid-layout-props

#     with dashboard.Grid(layout, draggableHandle=".draggable"):

#         # First card, the code editor.
#         #
#         # We use the 'key' parameter to identify the correct dashboard item.
#         #
#         # To make card's content automatically fill the height available, we will use CSS flexbox.
#         # sx is a parameter available with every Material UI widget to define CSS attributes.
#         #
#         # For more information regarding Card, flexbox and sx:
#         # https://mui.com/components/cards/
#         # https://mui.com/system/flexbox/
#         # https://mui.com/system/the-sx-prop/

#         with mui.Card(key="editor", sx={"display": "flex", "flexDirection": "column"}):

#             # To make this header draggable, we just need to set its classname to 'draggable',
#             # as defined above in dashboard.Grid's draggableHandle.

#             mui.CardHeader(title="Editor", className="draggable")

#             # We want to make card's content take all the height available by setting flex CSS value to 1.
#             # We also want card's content to shrink when the card is shrinked by setting minHeight to 0.

#             with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

#                 # Here is our Monaco code editor.
#                 #
#                 # First, we set the default value to st.session_state.data that we initialized above.
#                 # Second, we define the language to use, JSON here.
#                 #
#                 # Then, we want to retrieve changes made to editor's content.
#                 # By checking Monaco documentation, there is an onChange property that takes a function.
#                 # This function is called everytime a change is made, and the updated content value is passed in
#                 # the first parameter (cf. onChange: https://github.com/suren-atoyan/monaco-react#props)
#                 #
#                 # Streamlit Elements provide a special sync() function. This function creates a callback that will
#                 # automatically forward its parameters to Streamlit's session state items.
#                 #
#                 # Examples
#                 # --------
#                 # Create a callback that forwards its first parameter to a session state item called "data":
#                 # >>> editor.Monaco(onChange=sync("data"))
#                 # >>> print(st.session_state.data)
#                 #
#                 # Create a callback that forwards its second parameter to a session state item called "ev":
#                 # >>> editor.Monaco(onChange=sync(None, "ev"))
#                 # >>> print(st.session_state.ev)
#                 #
#                 # Create a callback that forwards both of its parameters to session state:
#                 # >>> editor.Monaco(onChange=sync("data", "ev"))
#                 # >>> print(st.session_state.data)
#                 # >>> print(st.session_state.ev)
#                 #
#                 # Now, there is an issue: onChange is called everytime a change is made, which means everytime
#                 # you type a single character, your entire Streamlit app will rerun.
#                 #
#                 # To avoid this issue, you can tell Streamlit Elements to wait for another event to occur
#                 # (like a button click) to send the updated data, by wrapping your callback with lazy().
#                 #
#                 # For more information on available parameters for Monaco:
#                 # https://github.com/suren-atoyan/monaco-react
#                 # https://microsoft.github.io/monaco-editor/api/interfaces/monaco.editor.IStandaloneEditorConstructionOptions.html

#                 editor.Monaco(
#                     defaultValue=st.session_state.data,
#                     language="json",
#                     onChange=lazy(sync("data"))
#                 )

#             with mui.CardActions:

#                 # Monaco editor has a lazy callback bound to onChange, which means that even if you change
#                 # Monaco's content, Streamlit won't be notified directly, thus won't reload everytime.
#                 # So we need another non-lazy event to trigger an update.
#                 #
#                 # The solution is to create a button that fires a callback on click.
#                 # Our callback doesn't need to do anything in particular. You can either create an empty
#                 # Python function, or use sync() with no argument.
#                 #
#                 # Now, everytime you will click that button, onClick callback will be fired, but every other
#                 # lazy callbacks that changed in the meantime will also be called.

#                 mui.Button("Apply changes", onClick=sync())

#         # Second card, the Nivo Bump chart.
#         # We will use the same flexbox configuration as the first card to auto adjust the content height.

#         with mui.Card(key="chart", sx={"display": "flex", "flexDirection": "column"}):

#             # To make this header draggable, we just need to set its classname to 'draggable',
#             # as defined above in dashboard.Grid's draggableHandle.

#             mui.CardHeader(title="Chart", className="draggable")

#             # Like above, we want to make our content grow and shrink as the user resizes the card,
#             # by setting flex to 1 and minHeight to 0.

#             with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

#                 # This is where we will draw our Bump chart.
#                 #
#                 # For this exercise, we can just adapt Nivo's example and make it work with Streamlit Elements.
#                 # Nivo's example is available in the 'code' tab there: https://nivo.rocks/bump/
#                 #
#                 # Data takes a dictionary as parameter, so we need to convert our JSON data from a string to
#                 # a Python dictionary first, with `json.loads()`.
#                 #
#                 # For more information regarding other available Nivo charts:
#                 # https://nivo.rocks/

#                 nivo.Bump(
#                     data=json.loads(st.session_state.data),
#                     colors={ "scheme": "spectral" },
#                     lineWidth=3,
#                     activeLineWidth=6,
#                     inactiveLineWidth=3,
#                     inactiveOpacity=0.15,
#                     pointSize=10,
#                     activePointSize=16,
#                     inactivePointSize=0,
#                     pointColor={ "theme": "background" },
#                     pointBorderWidth=3,
#                     activePointBorderWidth=3,
#                     pointBorderColor={ "from": "serie.color" },
#                     axisTop={
#                         "tickSize": 5,
#                         "tickPadding": 5,
#                         "tickRotation": 0,
#                         "legend": "",
#                         "legendPosition": "middle",
#                         "legendOffset": -36
#                     },
#                     axisBottom={
#                         "tickSize": 5,
#                         "tickPadding": 5,
#                         "tickRotation": 0,
#                         "legend": "",
#                         "legendPosition": "middle",
#                         "legendOffset": 32
#                     },
#                     axisLeft={
#                         "tickSize": 5,
#                         "tickPadding": 5,
#                         "tickRotation": 0,
#                         "legend": "ranking",
#                         "legendPosition": "middle",
#                         "legendOffset": -40
#                     },
#                     margin={ "top": 40, "right": 100, "bottom": 40, "left": 60 },
#                     axisRight=None,
#                 )

#         # Third element of the dashboard, the Media player.

#         with mui.Card(key="media", sx={"display": "flex", "flexDirection": "column"}):
#             mui.CardHeader(title="Media Player", className="draggable")
#             with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

#                 # This element is powered by ReactPlayer, it supports many more players other
#                 # than YouTube. You can check it out there: https://github.com/cookpete/react-player#props

#                 media.Player(url=media_url, width="100%", height="100%", controls=True)


# import streamlit as st
# import cv2
# import numpy as np

# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer with OpenCV:
#     bytes_data = img_file_buffer.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Check the type of cv2_img:
#     # Should output: <class 'numpy.ndarray'>
#     st.write(type(cv2_img))

#     # Check the shape of cv2_img:
#     # Should output shape: (height, width, channels)
#     st.write(cv2_img.shape)