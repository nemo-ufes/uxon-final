import stardog
import io
from werkzeug.datastructures import FileStorage
import csv

from flask import Flask, render_template, request, redirect, flash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from owlready2 import *

onto = get_ontology('https://raw.githubusercontent.com/sd-costa/HCI-ON-owl/main/uxon-hcion-extract.owl').load()

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['csv'])

result_interactivity = {}
result_behavior = {}

# connection_details = {
#     'endpoint': 'http://dev.nemo.inf.ufes.br:5820',
#     'username': 'admin',
#     'password': 'admin'
# }

connection_details = {
  'endpoint': 'http://localhost:5820',
  'username': 'admin',
  'password': 'admin'
}

conn = stardog.Connection('uxon', **connection_details)


def loadIndividuals(log_file):
    # Clean the Ontology
    conn.begin()
    conn.clear()
    conn.commit()

    for i in list(onto.individuals()):
        destroy_entity(i)

    log_file.pop(0)
    reader = csv.reader(log_file)
    sortedlist = sorted(reader, key=lambda row: row[5], reverse=False)
    flag = 0

    startTime = datetime.time()
    endTime = datetime.time()
    with onto:
        for row in sortedlist:
            try:
                user, x, y, z, time, hour, sound = row
                # Instanciando o usuario
                individual = onto.User(user)
                individual.user_id = int(user)

                # Instanciando a Participacao
                participation = onto.UserParticipation()
                participation.up_sound = int(sound)
                participation.up_geolocation_x = x
                participation.up_geolocation_y = y
                participation.up_geolocation_z = z

                # Separando as horas, minutos e segundos
                splited_hour = hour.split(':')

                # Obtem tempo de inicio e tempo de final da sessao
                if flag == 0:
                    startTime = datetime.time(int(splited_hour[0]), int(splited_hour[1]), int(splited_hour[2]))
                    flag = 1
                else:
                    endTime = datetime.time(int(splited_hour[0]), int(splited_hour[1]), int(splited_hour[2]))

                # Adicionando a hora na participacao
                participation.up_timestamp = datetime.time(int(splited_hour[0]), int(splited_hour[1]),
                                                           int(splited_hour[2]))

                # Relacionando usuario e participacao
                individual.participation_of.append(participation)
            except(ValueError):
                print("Removing blank line...")

    users = get_all_users()
    for u in users:
        interactivityMeasurement = onto.Measurement()
        behaviorMeasurement = onto.Measurement()
        interactivityMeasure = onto.Measure()
        behaviorMeasure = onto.Measure()

        interactivityMeasure.is_applied_by.append(interactivityMeasurement)
        behaviorMeasure.is_applied_by.append(behaviorMeasurement)

        humanInteraction = onto.HumanComputerInteraction()
        evaluation = onto.HCIEvaluation()
        evaluationReport = onto.HCIEvaluationReport()

        evaluation.evaluates.append(humanInteraction)
        evaluation.creates.append(evaluationReport)

        humanInteraction.hci_start_time = startTime
        humanInteraction.hci_end_time = endTime
        user_participations = get_participations_byUser(u[0])
        first_time = user_participations[0][0].up_timestamp
        last_time = user_participations[len(user_participations) - 1][0].up_timestamp

        time_elapsed = subtractDatetimes(first_time, last_time)
        humanInteraction.hci_pduration = (datetime.datetime.min + time_elapsed).time()

        countSound = 0
        countPosition = 0
        index = 0
        for p in user_participations:
            newP = p[0]
            humanInteraction.is_composed_of.append(p[0])
            p[0].is_measured_by.append(interactivityMeasurement)
            p[0].is_measured_by.append(behaviorMeasurement)
            if index == 0:
                countPosition = 1
                p[0].up_type = "geolocation"
            else:
                if newP.up_sound != oldP.up_sound:
                    countSound = countSound + 1
                    p[0].up_type = "sound"
                else:
                    countPosition = countPosition + 1
                    p[0].up_type = "geolocation"

            oldP = p[0]
            index = index + 1

        interactivityMeasure.m_formula = "Tsec = Tui/Tsg"
        tsg = (datetime.datetime.min + subtractDatetimes(startTime, endTime)).time()
        tsgFloat = tsg.minute + tsg.second / 60.0
        tuiFloat = humanInteraction.hci_pduration.minute + humanInteraction.hci_pduration.second / 60.0
        interactivityMeasurement.mt_measured_value = tuiFloat / tsgFloat
        result_interactivity[int(u[0].user_id)] = tuiFloat / tsgFloat

        behaviorMeasure.m_formula = "Scomp = (Pxy + Psom)"
        behaviorMeasurement.mt_measured_value = countSound + countPosition
        result_behavior[int(u[0].user_id)] = countSound + countPosition

    global fil
    fil = open('./static/teste.owl', 'w')
    onto.save(file='./static/teste.owl', format="rdfxml")

    with conn:
        conn.begin()
        conn.add(stardog.content.File('./static/teste.owl'))
        conn.commit()

    interactivityDf = measureUserEngagement2()
    max = interactivityDf['Engagement'].max()
    interactivityDf['Percentual'] = interactivityDf.apply(lambda row: make_percentual(row, max), axis=1)
    interactivityDf["Interactivity"] = 100 * interactivityDf["Interactivity"]
    print(interactivityDf)
    print(interactivityDf.dtypes)
    for u in users:
        user_participations = get_participations_byUser(u[0])
        percentualMeasurement = onto.Measurement()
        percentualMeasure = onto.Measure()
        percentualMeasure.is_applied_by.append(percentualMeasurement)

        percentualMeasure.m_formula = "Percentual of Interactions"
        value = interactivityDf.loc[interactivityDf['User'] == u[0].user_id]['Percentual'].values[0]
        percentualMeasurement.mt_measured_value = float(value)
        for p in user_participations:
            p[0].is_measured_by.append(percentualMeasurement)

    fil = open('./static/teste.owl', 'w')
    onto.save(file='./static/teste.owl', format="rdfxml")

    with conn:
        conn.begin()
        conn.clear()
        conn.add(stardog.content.File('./static/teste.owl'))
        conn.commit()


def make_percentual(row, max):
    value = 100 * row['Engagement'] / max
    return value


def createFirstGraph():
    interactivityDf = measureUserEngagement3()
    # max = interactivityDf['Engagement'].max()
    # interactivityDf['Percentual'] = interactivityDf.apply (lambda row: make_percentual(row,max), axis=1)
    interactivityDf["Interactivity"] = 100 * interactivityDf["Interactivity"]
    interactivityDf['Interactivity'] = interactivityDf['Interactivity'].map('{:,.2f}'.format)
    interactivityDf['Percentual'] = interactivityDf['Percentual'].map('{:,.2f}'.format)
    interactivityDf = interactivityDf.astype({'Interactivity': 'float64'})
    interactivityDf = interactivityDf.astype({'Percentual': 'float64'})

    interactivityDf = interactivityDf.astype({'User': 'str'})
    tableFig = go.Figure(data=[
        go.Bar(name='Interactivity', x=interactivityDf['User'], y=interactivityDf['Interactivity'],
               text=interactivityDf["Interactivity"]),
        go.Bar(name='Percentage of Interactions', x=interactivityDf['User'], y=interactivityDf['Percentual'],
               text=interactivityDf["Percentual"])
    ])
    tableFig.update_xaxes(type='category')
    tableFig.update_layout(title='User Interactivity and Percentage of Interactions')

    # tableFig.write_image(file="static/tableFig.png", format='png')
    tableFig.write_image(file="./static/tableFig.png", format='png')
    return tableFig.to_html(full_html=False), interactivityDf


@app.route('/measured-value')
def showMoreInteractivity():
    tableFigHtml, dataframeFull = createFirstGraph()

    engagementFig = px.bar(dataframeFull, x="User", y="Engagement", color="User", title="User Engagement",
                           text=dataframeFull["Engagement"])
    engagementFig.update_xaxes(type='category')
    engagementFig.write_image(file="./static/engagementFig.png", format='png')
    # engagementFig.write_image(file="static/engagementFig.png", format='png')
    engagementFigHtml = engagementFig.to_html(full_html=False)

    users = get_all_users2()

    all_participations_df = pd.DataFrame(columns=['pessoa', 'x', 'y', 'z', 'som', 'hora'])
    graphs = {}
    interactive_graphs = {}
    graphsImages = {}
    for u in users:
        # Saving Percentual of interactions
        percentualInteractionMeasurement = onto.Measurement()
        percentualInteractionMeasurement.mt_formula = "Percentual of Interactions"
        userInDf = dataframeFull['User'] == str(u)
        dfFiltered = dataframeFull[userInDf]

        percentualInteractionMeasurement.mt_measured_value = float(dfFiltered.iloc[0]['Percentual'])
        df = get_participations_byUser2(int(u))

        df["x"] = pd.to_numeric(df["x"], errors='coerce')
        df["y"] = pd.to_numeric(df["y"], errors='coerce')
        df["z"] = pd.to_numeric(df["z"], errors='coerce')

        df["Sound"] = df["som"];

        df = df.astype({'Sound': 'int64'})
        df = df.astype({'som': 'string'})
        # df = df.astype({'som': 'int64'})
        df = df.astype({'pessoa': 'int64'})
        df = df.astype({'x': 'float64'})
        df = df.astype({'y': 'float64'})

        fig = px.scatter(df, x=df.x, y=df.y, color=df.som, size=df.Sound, hover_data=[df.pessoa, df.hora],labels={"som": "Sound", "hora": "Time", "pessoa": "User"})

        # fig = px.scatter(df, x=df.x, y=df.y, color=df.som,error_x_minus=None, error_y_minus=None, range_x=[-1000, 1000], range_y=[-1000, 1000])

        fig_interactive = px.scatter(df, x=df.x, y=df.y, animation_frame=df.hora, animation_group=df.pessoa,
                                     size=df.Sound, color=df.Sound, hover_name=df.pessoa,
                                     error_x_minus=None, error_y_minus=None, range_x=[-1000, 1000],
                                     range_y=[-1000, 1000],labels={"som": "Sound", "hora": "Time", "pessoa": "User"})

        graphs[int(u)] = fig.to_html(full_html=False)
        graphsImages[int(u)] = fig.to_image(format='png')
        # fig.write_image(file="static/graph"+str(u)+".png", format='png')
        fig.write_image(file="./static/graph" + str(u) + ".png", format='png')
        interactive_graphs[int(u)] = fig_interactive.to_html(full_html=False, auto_play=False)

    all_participations_df = queryStardog2()

    all_participations_df["x"] = pd.to_numeric(all_participations_df["x"], errors='coerce')
    all_participations_df["y"] = pd.to_numeric(all_participations_df["y"], errors='coerce')
    all_participations_df["z"] = pd.to_numeric(all_participations_df["z"], errors='coerce')

    all_participations_df["Sound"] = all_participations_df["som"];

    all_participations_df = all_participations_df.astype({'Sound': 'int64'})
    all_participations_df = all_participations_df.astype({'som': 'string'})

    # all_participations_df = all_participations_df.astype({'som': 'int64'})
    all_participations_df = all_participations_df.astype({'pessoa': 'int64'})
    all_participations_df = all_participations_df.astype({'x': 'float64'})
    all_participations_df = all_participations_df.astype({'y': 'float64'})

    all_participations_df = all_participations_df.sort_values(by='hora')

    figAll = px.scatter(all_participations_df, x=all_participations_df.x, y=all_participations_df.y,
                        color=all_participations_df.som, size=all_participations_df.Sound,
                        hover_data=[all_participations_df.pessoa, all_participations_df.hora],labels={"som": "Sound", "hora": "Time", "pessoa": "User"})
    figAllInteractive = px.scatter(all_participations_df, x=all_participations_df.x, y=all_participations_df.y,
                                   animation_frame=all_participations_df.hora,
                                   animation_group=all_participations_df.pessoa,
                                   size=all_participations_df.Sound, color=all_participations_df.Sound,
                                   hover_name=all_participations_df.pessoa, error_x_minus=None, error_y_minus=None,
                                   range_x=[-1000, 1000], range_y=[-1000, 1000],labels={"som": "Sound", "hora": "Time", "pessoa": "User"})

    graphs[0] = figAll.to_html(full_html=False)
    graphsImages[0] = figAll.to_image(format='png')
    figAll.write_image(file="./static/graph0.png", format='png')
    # figAll.write_image(file="static/graph0.png", format='png')

    interactive_graphs[0] = figAllInteractive.to_html(full_html=False, auto_play=False)

    figTopSound, figTopUser = createTopGraphs()
    return render_template('interactivity.html', interactivity=dataframeFull.values.tolist(), fig=graphs,
                           figInteractive=interactive_graphs, tableFig=tableFigHtml, engagementFig=engagementFigHtml,
                           graphsImages=graphsImages, sizeFigs=len(graphs.keys()), keys=list(graphs.keys()),
                           figTopSound=figTopSound, figTopUser=figTopUser)


@app.route('/queries')
def definedQueries():
    allUsers = get_all_users2()

    all = {'user_id': 'All Users'}
    data = [{'user_id': user} for user in allUsers]
    data.append(all)

    allSounds = get_all_sounds2()
    # data2 = [sound[0] for sound in allSounds]
    allSounds.sort()

    return render_template('respostas.html', interactivity=[], data=data, data2=allSounds, selectValue=None, usage=[],
                           selectValue2=None, type=0, fig={}, fig2={})


def createTopGraphs():
    topSound = get_top_5_sound()
    topUser = get_top_5_user()

    topSound = topSound.astype({'Sound': 'str'})
    tableFig = px.bar(topSound, x="Sound", y="Usage", color="Usage", title="Top 5 Sound Usage", text=topSound["Usage"])
    tableFig.update_xaxes(type='category')

    topUser = topUser.astype({'User': 'str'})
    tableFig2 = px.bar(topUser, x="User", y="Participations", color="Participations", title="Top 5 Users",
                       text=topUser["Participations"])
    tableFig2.update_xaxes(type='category')

    tableFig.write_image(file="./static/topSound.png", format='png')
    tableFig2.write_image(file="./static/topUsers.png", format='png')
    # tableFig.write_image(file="static/topSound.png", format='png')
    # tableFig2.write_image(file="static/topUsers.png", format='png')
    return tableFig.to_html(full_html=False), tableFig2.to_html(full_html=False)


def createGraphs(user, participations_list):
    df = pd.DataFrame(columns=['pessoa', 'x', 'y', 'z', 'som', 'hora'])
    for participationItem in participations_list:
        try:
            new_row = {'pessoa': int(user.user_id), 'x': float(participationItem.up_geolocation_x),
                       'y': float(participationItem.up_geolocation_y), 'z': float(participationItem.up_geolocation_z),
                       'som': int(participationItem.up_sound), 'hora': str(participationItem.up_timestamp)}
            df = df.append(new_row, ignore_index=True)
        except ValueError:
            print("Removing string cordinates from graph")

    return df


# def createAllGraphs(dataFrame):
#     users = get_all_users()
#     print("users")
#     print(users)
#     graphs = {}
#
#     for u in users:
#         user_id = int(u[0].userId)
#         filter = dataFrame[dataFrame['User'] == user_id]
#         fig = px.scatter(filter, x=filter.x, y=filter.y, color=filter.Sound)
#         graphs[user_id] = fig.to_html(full_html=False)
#
#     return graphs

@app.route("/queries", methods=['POST'])
def querySoundChange():
    requestForm = request
    select = requestForm.form.get('select')
    select2 = requestForm.form.get('select2')
    type = requestForm.form.get('type_query')
    user = None

    query2 = []
    filter = {}
    usage = {}
    selectedValue = None
    selectedValue2 = None

    allSoundInteractions = get_all_participation_type_sound2()

    allSoundInteractions = allSoundInteractions.astype({'Sound': 'int64'})
    allSoundInteractions = allSoundInteractions.astype({'User': 'int64'})

    allUsers = get_all_users2()

    all = {'user_id': "All Users"}
    data = [{'user_id': user} for user in allUsers]
    data.append(all)

    # First Query
    if select == 'All Users':
        filter = allSoundInteractions.copy()
        filter = filter.sort_values(by='Time')
        selectedValue = "All Users"
    else:
        if select != None:
            id = int(select)
            filter = allSoundInteractions[allSoundInteractions['User'] == id].copy()
            selectedValue = select

    allSounds = get_all_sounds2()
    # data2 = [sound[0] for sound in allSounds]
    allSounds.sort()

    # Second Query
    if select2 != None:
        sound = int(select2)
        query2 = allSoundInteractions[allSoundInteractions['Sound'] == sound].copy()
        usage = query2['User'].value_counts().to_frame()
        selectedValue2 = sound

    graphs = {}
    if (type == '2'):
        if (select == 'All Users'):
            filter = filter.astype({'Sound': 'str'})
            filter = filter.astype({'User': 'str'})
            graphs[0] = px.scatter(filter, x=filter.x, y=filter.y, color=filter.Sound, error_x_minus=None,
                                   error_y_minus=None, range_x=[-1000, 1000], range_y=[-1000, 1000]).to_html(
                full_html=False)
        else:
            filter = filter.astype({'Sound': 'str'})
            filter = filter.astype({'User': 'str'})
            fig = px.scatter(filter, x=filter.x, y=filter.y, color=filter.Sound, error_x_minus=None, error_y_minus=None,
                             range_x=[-1000, 1000], range_y=[-1000, 1000])
            graphs[id] = fig.to_html(full_html=False)

    graphs2 = {}
    if (type == '3'):
        query2 = query2.astype({'Sound': 'str'})
        query2 = query2.astype({'User': 'str'})
        fig = px.scatter(query2, x=query2.x, y=query2.y, color=query2.User, error_x_minus=None, error_y_minus=None,
                         range_x=[-1000, 1000], range_y=[-1000, 1000])
        graphs2[0] = fig.to_html(full_html=False)

    return render_template('respostas.html', interactivity=filter, data=data, data2=allSounds,
                           selectValue=selectedValue,
                           usage=usage, selectValue2=selectedValue2, type=type, fig=graphs, fig2=graphs2)


@app.route("/custom-queries")
def customQueries():
    allUsers = get_all_users2()

    all = {'user_id': 'All Users'}
    data = [{'user_id': user} for user in allUsers]
    data.insert(0, all)

    allSounds = get_all_sounds2()

    all2 = "All Sounds"
    # data2 = [sound[0] for sound in allSounds]
    allSounds.sort()
    allSounds.insert(0, all2)

    return render_template("custom.html", query=[], inputValue='', inputValue2='', users=data, sounds=allSounds,
                           user=None,
                           sound=None, typeS=None, user2=None, typeS2=None, inputValue3='', inputValue4='', type=0)


@app.route("/custom-queries", methods=['POST'])
def customQuery():
    requestForm = request

    type = requestForm.form.get('type_query')

    input1 = requestForm.form.get('input1')
    input2 = requestForm.form.get('input2')

    userSelect = requestForm.form.get('selectUser')
    soundSelect = requestForm.form.get('selectSound')
    typeSelect = requestForm.form.get('selectType')

    filters = [userSelect, soundSelect, typeSelect, input1, input2]
    filterBool = [False for _ in range(5)]

    for index, value in enumerate(filters):
        if value != '' and value is not None:
            filterBool[index] = True

    if int(type) == 1 or (True in filterBool):
        query1 = get_all_participations_filter2(filters)
    else:
        query1 = []

    input3 = requestForm.form.get('input3')
    input4 = requestForm.form.get('input4')

    userSelect2 = requestForm.form.get('selectUser2')
    typeSelect2 = requestForm.form.get('selectType2')

    if input3 != '':
        input3Int = int(input3)
    else:
        input3Int = None

    if input4 != '':
        input4Int = int(input4)
    else:
        input4Int = None

    filters2 = [userSelect2, typeSelect2, input3Int, input4Int]
    filterBool2 = [False for _ in range(4)]

    for index, value in enumerate(filters2):
        if value != '' and value is not None:
            filterBool2[index] = True

    if int(type) == 2 or (True in filterBool2):
        if True not in filterBool2:
            query2 = get_all_number_participations()
        else:
            # search with filter
            query2 = get_all_number_participations_filter2(filters2)
    else:
        query2 = []

    allUsers = get_all_users2()

    all = {'user_id': 'All Users'}
    data = [{'user_id': user} for user in allUsers]
    data.append(all)

    allSounds = get_all_sounds2()
    all2 = "All Sounds"
    # data2 = [sound[0] for sound in allSounds]
    allSounds.sort()
    allSounds.insert(0, all2)

    return render_template('custom.html', query=query1, query2=query2, inputValue=input1, inputValue2=input2,
                           users=data, sounds=allSounds, user=userSelect, sound=soundSelect, typeS=typeSelect,
                           user2=userSelect2, typeS2=typeSelect2, inputValue3=input3, inputValue4=input4, type=type)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/evaluation-report', methods=['POST'])
def evaluation_report():
    requestForm = request

    comment = requestForm.form.get('comment')

    print(comment)
    return ("nothing")


@app.route('/')
def index():
    return render_template('inicio.html')


@app.route("/", methods=["POST"])
def upload_image():
    if 'file[0]' not in request.files:
        flash('No file selected. Please choose a file.')
        return redirect(request.url)
    file = request.files['file[0]']

    # print('tipo do arquivo ' + type(file))
    if file.filename == '':
        flash('No file selected. Please choose a file.')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        decoded_file = file.read().decode('utf-8').splitlines()
        inicio = time.time()
        loadIndividuals(decoded_file)
        fim = time.time()
        print(fim - inicio)
    else:
        flash('Only CSV files allowed.')
        return redirect(request.url)

    return redirect('/measured-value')


def get_all_users():
    return list(default_world.sparql("""
                     SELECT ?usuario
                    { ?usuario rdf:type uxon-hcion-extract:User .
                    }
                """))


def get_all_users2():
    query = """
      SELECT ?UserId where
        {
          ?user rdf:type uxon:User .
          ?user uxon:user_id ?UserId .
        }
      """

    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return list(df["UserId"])


def get_participations_byUser(user):
    return list(default_world.sparql("""
                         SELECT ?individual
                        {
                           ?? uxon-hcion-extract:participation_of ?individual .
                           ?individual uxon-hcion-extract:up_timestamp ?hour .
                         } ORDER BY ?hour
                    """, [user]))


def subtractDatetimes(dt1, dt2):
    date = datetime.date(1, 1, 1)
    datetime1 = datetime.datetime.combine(date, dt1)
    datetime2 = datetime.datetime.combine(date, dt2)
    return datetime2 - datetime1


# def measureUserEngagement():
#     return list(default_world.sparql("""
#                          SELECT ?userId ?interactivityValue ?behaviorValue
#                         {
#                            ?individual uxon-hcion-extract:participation_of ?part
#                            ?individual uxon-hcion-extract:user_id ?userId
#                            ?part uxon-hcion-extract:is_measured_by ?interactivityMeasurement .
#                            ?interactivityMeasurement uxon-hcion-extract:mt_measured_value ?interactivityValue
#                            ?interactivityMeasurement uxon-hcion-extract:mt_formula "Tsec = Tui/Tsg"
#                            ?part uxon-hcion-extract:is_measured_by ?behaviorMeasurement .
#                            ?behaviorMeasurement uxon-hcion-extract:mt_measured_value ?behaviorValue
#                            ?behaviorMeasurement uxon-hcion-extract:mt_formula "Scomp = (Pxy + Psom)"
#                          } GROUP BY ?individual ?interactivityValue ?behaviorValue
#                     """))


# def get_all_sounds():
#     return list(default_world.sparql("""
#                         SELECT DISTINCT ?sound
#                         {
#                             ?sound_participation rdf:type uxon-hcion-extract:UserParticipation .
#                             ?sound_participation uxon-hcion-extract:up_type "sound" .
#                             ?sound_participation uxon-hcion-extract:up_sound ?sound
#                          }
#                         """))

def get_all_sounds2():
    query = """
        SELECT DISTINCT ?Sound where
          {
            ?sound_participation rdf:type uxon:UserParticipation .
            ?sound_participation uxon:up_type "sound" .
            ?sound_participation uxon:up_sound ?Sound
          }
        """

    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return list(df["Sound"])


def get_all_participation_type_sound():
    return list(default_world.sparql("""
                                    SELECT ?userId ?x ?y ?sound ?hour
                                    {
                                        ?sound_participation rdf:type uxon-hcion-extract:UserParticipation .
                                        ?user uxon-hcion-extract:participation_of ?sound_participation .
                                        ?user uxon-hcion-extract:user_id ?userId .
                                        ?sound_participation uxon-hcion-extract:up_geolocation_x ?x .
                                        ?sound_participation uxon-hcion-extract:up_geolocation_y ?y .
                                        ?sound_participation uxon-hcion-extract:up_timestamp ?hour .
                                        ?sound_participation uxon-hcion-extract:up_sound ?sound .
                                        ?sound_participation uxon-hcion-extract:up_type "sound" .
                                    } ORDER BY ?hour
                              """))


def get_all_participation_type_sound2():
    query = """
      SELECT ?User ?x ?y ?Sound ?Time where
        {
          ?sound_participation rdf:type uxon:UserParticipation .
          ?user uxon:participation_of ?sound_participation .
          ?user uxon:user_id ?User .
          ?sound_participation uxon:up_geolocation_x ?x .
          ?sound_participation uxon:up_geolocation_y ?y .
          ?sound_participation uxon:up_timestamp ?Time .
          ?sound_participation uxon:up_sound ?Sound .
          ?sound_participation uxon:up_type "sound" .
        } ORDER BY ?Time
      """

    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def get_all_participations_filter2(filters):
    query = """
            SELECT ?user ?sound ?type ?time where {
                ?participation rdf:type uxon:UserParticipation .
                ?participation uxon:up_timestamp ?time .
                ?userObj uxon:participation_of ?participation .
                ?userObj uxon:user_id ?user .
                ?participation uxon:up_type ?type .
                ?participation uxon:up_sound ?sound .
    """
    # params = []

    for index, filter in enumerate(filters):
        if filter:
            # user
            if index == 0 and filters[index] != 'All Users':
                query = query + """?userObj uxon:user_id %d .
                """ % int(filters[index])
                # params.append(int(filters[index]))
            # sound
            if index == 1 and filters[index] != 'All Sounds':
                query = query + """?participation uxon:up_sound %d .
                """ % int(filters[index])
                # params.append(int(filters[index]))
            # type
            if index == 2 and int(filters[index]) != 0:
                if int(filters[index]) == 1:
                    type = 'sound'
                if int(filters[index]) == 2:
                    type = 'geolocation'

                query = query + """?participation uxon:up_type '%s' .
                """ % type
                # params.append(type)
            # input1
            if index == 3:
                # params.append(filters[index])
                if (filters[4] != ''):
                    query = query + """FILTER(?time >= '%s' && ?time <= '%s').""" % (filters[index], filters[4])
                    # params.append(filters[4])
                else:
                    query = query + """FILTER(?time >= '%s').""" % filters[index]
            if index == 4:
                if (filters[3] == ''):
                    query = query + """FILTER(?time <= '%s').""" % filters[index]
                    # params.append(filters[index])
    query = query + """
    } ORDER BY ?time
    """
    print(query)
    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    print(df.head())
    return df
    # return list(default_world.sparql(select, params))


# def get_all_participations_filter(filters):
#     select = """
#             SELECT ?user ?sound ?type ?time {
#                 ?participation rdf:type uxon-hcion-extract:UserParticipation .
#                 ?participation uxon-hcion-extract:up_timestamp ?time .
#                 ?userObj uxon-hcion-extract:participation_of ?participation .
#                 ?userObj uxon-hcion-extract:user_id ?user .
#                 ?participation uxon-hcion-extract:up_type ?type .
#                 ?participation uxon-hcion-extract:up_sound ?sound .
#     """
#     params = []
#
#     for index, filter in enumerate(filters):
#         if filter:
#             # user
#             if index == 0 and filters[index] != 'All Users':
#                 select = select + """?userObj uxon-hcion-extract:user_id ?? .
#                 """
#                 params.append(int(filters[index]))
#             # sound
#             if index == 1 and filters[index] != 'All Sounds':
#                 select = select + """?participation uxon-hcion-extract:up_sound ?? .
#                 """
#                 params.append(int(filters[index]))
#             # type
#             if index == 2 and int(filters[index]) != 0:
#                 if int(filters[index]) == 1:
#                     type = 'sound'
#                 if int(filters[index]) == 2:
#                     type = 'geolocation'
#
#                 select = select + """?participation uxon-hcion-extract:up_type ?? .
#                 """
#                 params.append(type)
#             # input1
#             if index == 3:
#                 params.append(filters[index])
#                 if (filters[4] != ''):
#                     select = select + """FILTER(?time >= ?? && ?time <= ??)."""
#                     params.append(filters[4])
#                 else:
#                     select = select + """FILTER(?time >= ??)."""
#             if index == 4:
#                 if (filters[3] == ''):
#                     select = select + """FILTER(?time <= ??)."""
#                     params.append(filters[index])
#     select = select + """
#     } ORDER BY ?time
#     """
#     return list(default_world.sparql(select, params))


def get_all_number_participations():
    return list(default_world.sparql("""
                     SELECT ?user (COUNT(?participation) as ?count) ?type
                    {   ?participation rdf:type uxon-hcion-extract:UserParticipation .
                        ?userObj uxon-hcion-extract:participation_of ?participation .
                        ?userObj uxon-hcion-extract:user_id ?user .
                        ?participation uxon-hcion-extract:up_type ?type .
                        ?participation uxon-hcion-extract:up_timestamp ?hour .
                    } GROUP BY ?user ?type
                """))


def get_all_number_participations_filter2(filters):
    select = """
             SELECT ?user (COUNT(?participation) as ?count) ?type where
                    {   ?participation rdf:type uxon:UserParticipation .
                        ?userObj uxon:participation_of ?participation .
                        ?userObj uxon:user_id ?user .
                        ?participation uxon:up_type ?type .
                        ?participation uxon:up_timestamp ?hour .
    """
    # params = []

    for index, filter in enumerate(filters):
        if filter:
            # user
            if index == 0 and filters[index] != 'All Users':
                select = select + """?userObj uxon:user_id %d .
                """ % int(filters[index])
                # params.append(int(filters[index]))
            # type
            if index == 1 and int(filters[index]) != 0:
                if int(filters[index]) == 1:
                    type = 'sound'
                if int(filters[index]) == 2:
                    type = 'geolocation'

                select = select + """?participation uxon:up_type '%s' .
                """ % type
                # params.append(type)
            # input3
            if index == 2:
                # params.append(filters[index])
                select = select + """
                     } GROUP BY ?user ?type
                    """
                if (filters[index + 1] != None):
                    select = select + """HAVING(?count >= %d && ?count <= %d)""" % (filters[index], filters[index + 1])
                    # params.append(filters[index + 1])
                else:
                    select = select + """HAVING(?count >= ??)""" % filters[index]
                    # params.append(filters[index])
            # input4
            if index == 3:
                if (filters[index - 1] == None):
                    select = select + """
                         } GROUP BY ?user ?type
                        """
                    select = select + """HAVING(?count <= %d)""" % filters[index]
                    # params.append(filters[index])

    if filters[2] is None and filters[3] is None:
        select = select + """
         } GROUP BY ?user ?type
        """
    csv_results = conn.select(select, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    print(df.head())
    return df


# def get_all_number_participations_filter(filters):
#     select = """
#              SELECT ?user (COUNT(?participation) as ?count) ?type
#                     {   ?participation rdf:type uxon-hcion-extract:UserParticipation .
#                         ?userObj uxon-hcion-extract:participation_of ?participation .
#                         ?userObj uxon-hcion-extract:user_id ?user .
#                         ?participation uxon-hcion-extract:up_type ?type .
#                         ?participation uxon-hcion-extract:up_timestamp ?hour .
#     """
#     params = []
#
#     for index, filter in enumerate(filters):
#         if filter:
#             # user
#             if index == 0 and filters[index] != 'All Users':
#                 select = select + """?userObj uxon-hcion-extract:user_id ?? .
#                 """
#                 params.append(int(filters[index]))
#             # type
#             if index == 1 and int(filters[index]) != 0:
#                 if int(filters[index]) == 1:
#                     type = 'sound'
#                 if int(filters[index]) == 2:
#                     type = 'geolocation'
#
#                 select = select + """?participation uxon-hcion-extract:up_type ?? .
#                 """
#                 params.append(type)
#             # input3
#             if index == 2:
#                 params.append(filters[index])
#                 select = select + """
#                      } GROUP BY ?user ?type
#                     """
#                 if (filters[index + 1] != None):
#                     select = select + """HAVING(?count >= ?? && ?count <= ??)"""
#                     params.append(filters[index + 1])
#                 else:
#                     select = select + """HAVING(?count >= ??)"""
#                     # params.append(filters[index])
#             # input4
#             if index == 3:
#                 if (filters[index - 1] == None):
#                     select = select + """
#                          } GROUP BY ?user ?type
#                         """
#                     select = select + """HAVING(?count <= ??)"""
#                     params.append(filters[index])
#
#     if filters[2] is None and filters[3] is None:
#         select = select + """
#          } GROUP BY ?user ?type
#         """
#     return list(default_world.sparql(select, params))


def get_participations_byUser2(user):
    query = """
  PREFIX uxon: <http://nemo.ufes.br/hcion-extract#>
  SELECT ?pessoa ?x ?y ?z ?som ?hora WHERE {
    ?individual uxon:up_timestamp ?hora .
    ?user uxon:participation_of ?individual .
    ?user uxon:user_id %d .
    ?user uxon:user_id ?pessoa .
    ?individual uxon:up_geolocation_x ?x .
    ?individual uxon:up_geolocation_y ?y .
    ?individual uxon:up_geolocation_z ?z .
    ?individual uxon:up_sound ?som .
  } ORDER BY ?hora
  """ % user

    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def queryStardog2():
    query = """
  PREFIX uxon: <http://nemo.ufes.br/hcion-extract#>
  SELECT ?pessoa ?x ?y ?z ?som ?hora WHERE {
    ?individual uxon:up_timestamp ?hora .
    ?user uxon:participation_of ?individual .
    ?user uxon:user_id ?pessoa .
    ?individual uxon:up_geolocation_x ?x .
    ?individual uxon:up_geolocation_y ?y .
    ?individual uxon:up_geolocation_z ?z .
    ?individual uxon:up_sound ?som .
  } ORDER BY ?hora
  """

    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def measureUserEngagement2():
    query = """
  PREFIX uxon: <http://nemo.ufes.br/hcion-extract#>
  SELECT ?User ?Interactivity ?Engagement WHERE {
    ?individual uxon:participation_of ?part .
    ?individual uxon:user_id ?User .
    ?part uxon:is_measured_by ?interactivityMeasurement .
    ?interactivityMeasurement uxon:mt_measured_value ?Interactivity .
    ?interactivityMeasure uxon:is_applied_by ?interactivityMeasurement .
    ?interactivityMeasure uxon:m_formula "Tsec = Tui/Tsg" .
    ?part uxon:is_measured_by ?behaviorMeasurement .
    ?behaviorMeasurement uxon:mt_measured_value ?Engagement .
    ?behaviorMeasure uxon:is_applied_by ?behaviorMeasurement .
    ?behaviorMeasure uxon:m_formula "Scomp = (Pxy + Psom)" .
  } GROUP BY ?User ?Interactivity ?Engagement
  """
    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def measureUserEngagement3():
    query = """
  PREFIX uxon: <http://nemo.ufes.br/hcion-extract#>
  SELECT ?User ?Interactivity ?Engagement ?Percentual WHERE {
    ?individual uxon:participation_of ?part .
    ?individual uxon:user_id ?User .
    ?part uxon:is_measured_by ?interactivityMeasurement .
    ?interactivityMeasurement uxon:mt_measured_value ?Interactivity .
    ?interactivityMeasure uxon:is_applied_by ?interactivityMeasurement .
    ?interactivityMeasure uxon:m_formula "Tsec = Tui/Tsg" .
    ?part uxon:is_measured_by ?behaviorMeasurement .
    ?behaviorMeasurement uxon:mt_measured_value ?Engagement .
    ?behaviorMeasure uxon:is_applied_by ?behaviorMeasurement .
    ?behaviorMeasure uxon:m_formula "Scomp = (Pxy + Psom)" .
    ?part uxon:is_measured_by ?percentualMeasurement .
    ?percentualMeasurement uxon:mt_measured_value ?Percentual .
    ?percentualMeasure uxon:is_applied_by ?percentualMeasurement .
    ?percentualMeasure uxon:m_formula "Percentual of Interactions" .
  } GROUP BY ?User ?Interactivity ?Engagement ?Percentual
  """
    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def get_top_5_sound():
    query = """
      select ?Sound (count(?Sound) as ?Usage) where {
        ?sound_participation rdf:type uxon:UserParticipation .
        ?sound_participation uxon:up_sound ?Sound .
        } group by ?Sound
        order by desc(?Usage)
        limit 5
    """
    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def get_top_5_user():
    query = """
        select ?User (count(?part) as ?Participations) where {
            ?individual uxon:participation_of ?part .
            ?individual uxon:user_id ?User .
        } group by ?User
        order by desc(?Participations)
        limit 5
  """
    csv_results = conn.select(query, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


# se quisermos especificar uma rota e uma porta
# Observação: não utilizar estas definições para produção,
# estas opções foram preparadas para ajudar no ambiente de desenvolvimento.
# app.run(host='0.0.0.0', port=8080)
if __name__ == '__main__':
    app.run(debug="False")