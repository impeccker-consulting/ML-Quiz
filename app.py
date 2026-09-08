import streamlit as st
from pathlib import Path
import csv, hashlib, json, io
from datetime import datetime, timezone
import shutil
from validate_submission import validate_submission
from evaluate_submission import evaluate_submission
from faculty_scoring import score_submission
from quiz.engine import load_bank, create_attempt, save_attempt, load_attempt, submit_attempt, is_expired, parse_iso
from quiz.access import verify_access_code, access_code
from quiz.reporting import build_reports, discuss_in_class, write_csv
FACULTY_KEY = st.secrets["FACULTY_KEY"]

BASE = Path(__file__).parent
DATA_DIR, SUB_DIR, RESULTS_DIR = BASE/'datasets', BASE/'submissions', BASE/'results'
for d in (DATA_DIR, SUB_DIR, RESULTS_DIR): d.mkdir(exist_ok=True)

ALLOCATION_FILE = BASE / 'student_dataset_allocation.csv'

DATASET_MAP = {}

if ALLOCATION_FILE.exists():

    with open(
        ALLOCATION_FILE,
        newline='',
        encoding='utf-8'
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            student = row.get('SAP ID', '').strip().upper()
            dataset = row.get('Assigned_Dataset', '').strip()

            if student and dataset:
                DATASET_MAP[student] = dataset
DATASET_META = {
    'ML_Dataset_01.csv': {'title': 'Individual ML Case', 'type': 'Regression'},
    'ML_Dataset_02.csv': {'title': 'Individual ML Case', 'type': 'Classification'},
}

st.set_page_config(page_title='ML Individual Assignment', page_icon='📊', layout='centered')
st.title('📊 Machine Learning Individual Assignment')
st.caption('MBA – Digital Transformation')

st.divider()
QUIZ_DIR = BASE / 'quiz'
QUIZ_BANK_FILE = QUIZ_DIR / 'question_bank.json'
QUIZ_CONFIG_FILE = QUIZ_DIR / 'config.json'

def _load_quiz_config():
    with open(QUIZ_CONFIG_FILE, encoding='utf-8') as f:
        return json.load(f)

def _quiz_window_status(config):
    now = datetime.now(timezone.utc)
    start = parse_iso(config.get('Start_DateTime'))
    end = parse_iso(config.get('End_DateTime'))
    if start and now < start:
        return 'NOT_STARTED'
    if end and now > end:
        return 'CLOSED'
    return 'OPEN'

def _quiz_student_view(sap_id):
    config = _load_quiz_config()

    if sap_id not in DATASET_MAP:
        st.error('SAP ID is not present in the student allocation list.')
        st.stop()

    if config.get('Access_Code_Required', True):
        access = st.text_input(
            'Quiz access code',
            type='password',
            max_chars=16,
            help='Use the access code issued for your SAP ID and this quiz.'
        ).strip().upper()
        if not access:
            st.info('Enter your quiz access code to continue.')
            st.stop()
        if not verify_access_code(sap_id, config['Quiz_ID'], access, FACULTY_KEY):
            st.error('Invalid quiz access code for this SAP ID.')
            st.stop()

    if config.get('Status') != 'ACTIVE':
        st.info('The quiz is not currently active.')
        st.caption(f"Quiz status: {config.get('Status', 'UNKNOWN')}")
        st.stop()

    window_status = _quiz_window_status(config)
    if window_status == 'NOT_STARTED':
        st.warning('The quiz has not started yet.')
        st.stop()
    if window_status == 'CLOSED':
        st.warning('The quiz is closed.')
        st.stop()

    bank = load_bank(QUIZ_BANK_FILE)
    attempt = load_attempt(QUIZ_DIR, config['Quiz_ID'], sap_id)

    if attempt is None:
        st.subheader(config['Quiz_Title'])
        st.info(
            f"20 questions • 20 marks • {config['Duration_Minutes']} minutes\n\n"
            "Your paper is generated deterministically from your SAP ID and Quiz ID. "
            "Refreshing the page will not generate a new paper."
        )
        if st.button('Start Quiz', type='primary'):
            try:
                attempt = create_attempt(QUIZ_DIR, bank, config, sap_id)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            st.session_state['quiz_attempt_started'] = True
            st.rerun()
        st.stop()

    if attempt['Submitted']:
        st.success('Your quiz has already been submitted.')
        # Score intentionally hidden from students.
        if attempt.get('Auto_Submitted'):
            st.caption('The quiz was automatically submitted when the time expired.')
        st.stop()

    if is_expired(attempt):
        submit_attempt(QUIZ_DIR, attempt, auto=True)
        st.rerun()

    # Server-side authoritative timer.
    # Remaining time is calculated from the persisted UTC deadline.
    deadline = parse_iso(attempt['Deadline_Timestamp'])

    @st.fragment(run_every='1s')
    def _quiz_timer():
        now = datetime.now(timezone.utc)
        remaining = max(0, int((deadline - now).total_seconds()))
        mins, secs = divmod(remaining, 60)

        timer_html = (
            '<div style="position:fixed;top:4.5rem;left:0;right:0;'
            'z-index:9999;background:#ffffff;text-align:center;'
            'padding:0.65rem 1rem 0.75rem 1rem;'
            'border-bottom:2px solid #444;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
            '<div style="font-size:1rem;font-weight:800;letter-spacing:1px;">'
            'TIME REMAINING</div>'
            '<div style="font-size:3rem;font-weight:900;'
            'line-height:1.1;letter-spacing:1px;">'
            f'{mins:02d}:{secs:02d}'
            '</div></div>'
        )
        st.markdown(timer_html, unsafe_allow_html=True)

        if remaining <= 0:
            fresh = load_attempt(QUIZ_DIR, config['Quiz_ID'], sap_id)
            if fresh and not fresh['Submitted']:
                submit_attempt(QUIZ_DIR, fresh, auto=True)
                st.rerun()

    _quiz_timer()

    # Reserve space for the viewport-fixed timer.
    st.markdown(
        '<div style="height:125px;"></div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(config['Quiz_Title'])
    st.caption(
        'Multi-response questions do not reveal how many answers are correct. '
        'Choose all statements you consider reasonable where instructed.'
    )

    # All 20 questions are rendered in the same stable order.
    # This makes Shown_Timestamp correspond to the presentation of the paper.
    changed = False

    for i, q in enumerate(attempt['Questions'], start=1):
        if q['Shown_Timestamp'] is None:
            q['Shown_Timestamp'] = datetime.now(timezone.utc).isoformat()
            changed = True

        st.markdown(f"### Question {i}")
        st.write(q['Question_Text'])

        option_map = {o['Option_ID']: o['Text'] for o in q['Options']}
        labels = [f"{oid}. {text}" for oid, text in option_map.items()]
        key = f"quiz_{config['Quiz_ID']}_{q['Question_ID']}"

        if q['Response_Type'] == 'Multi':
            previous = q.get('Final_Response') or []
            response = st.multiselect(
                'Select your response(s)',
                labels,
                key=key,
                default=[
                    f"{oid}. {option_map[oid]}"
                    for oid in previous
                    if oid in option_map
                ],
                label_visibility='collapsed'
            )
            current_ids = [
                label.split('.', 1)[0]
                for label in response
            ]
        else:
            previous = q.get('Final_Response')
            if isinstance(previous, list):
                previous = previous[0] if previous else None
            default_index = 0
            if previous in option_map:
                default_index = list(option_map).index(previous) + 1
            choices = [''] + labels
            response = st.radio(
                'Select one response',
                choices,
                index=default_index,
                key=key,
                label_visibility='collapsed'
            )
            current_ids = [] if not response else [response.split('.', 1)[0]]

        old_ids = q.get('Final_Response')
        old_ids = [] if old_ids is None else (
            old_ids if isinstance(old_ids, list) else [old_ids]
        )

        if current_ids != old_ids:
            now = datetime.now(timezone.utc).isoformat()
            if current_ids and q['First_Response_Timestamp'] is None:
                q['First_Response_Timestamp'] = now
            if old_ids:
                q['Answer_Changed'] = True
            q['Final_Response_Timestamp'] = now
            q['Final_Response'] = current_ids
            q['Answered'] = bool(current_ids)
            changed = True

        st.divider()

    if changed:
        save_attempt(QUIZ_DIR, attempt)

    answered = sum(bool(q.get('Final_Response')) for q in attempt['Questions'])
    st.write(f"**Answered: {answered} / 20**")

    if st.button('Submit Quiz', type='primary'):
        fresh = load_attempt(QUIZ_DIR, config['Quiz_ID'], sap_id)
        if fresh is None:
            st.error('Quiz attempt could not be found.')
            st.stop()
        if is_expired(fresh):
            submit_attempt(QUIZ_DIR, fresh, auto=True)
        else:
            submit_attempt(QUIZ_DIR, fresh, auto=False)
        st.rerun()


faculty_mode = st.checkbox('Faculty Mode')

if faculty_mode:

    faculty_key = st.text_input(
        'Faculty access key',
        type='password'
    )

    if faculty_key != FACULTY_KEY:
        st.info('Faculty access required.')
        st.stop()

if faculty_mode:

    faculty_section = st.radio(
        'Faculty section',
        ['Assignment Evaluation', 'Quiz Report'],
        horizontal=True
    )

    if faculty_section == 'Quiz Report':
        quiz_config = _load_quiz_config()
        quiz_id = quiz_config['Quiz_ID']

        st.subheader('Quiz Report')

        # -------------------------------------------------
        # Faculty-only quiz access-code export
        # -------------------------------------------------

        access_rows = []

        for sap_id in sorted(DATASET_MAP):
            access_rows.append({
                'SAP ID': sap_id,
                'Quiz ID': quiz_id,
                'Access Code': access_code(
                    sap_id,
                    quiz_id,
                    FACULTY_KEY
                )
            })

        access_buffer = io.StringIO()
        writer = csv.DictWriter(
            access_buffer,
            fieldnames=['SAP ID', 'Quiz ID', 'Access Code']
        )
        writer.writeheader()
        writer.writerows(access_rows)

        st.markdown('**Student Quiz Access Codes**')
        st.caption(
            'Faculty-only export. Each code is tied to the SAP ID and current Quiz ID.'
        )

        st.download_button(
            'Download Quiz Access Codes CSV',
            access_buffer.getvalue().encode('utf-8-sig'),
            'ML_QUIZ_01_access_codes.csv',
            'text/csv',
            key='quiz_access_codes_export'
        )

        reports = build_reports(BASE / 'quiz', quiz_id)

        if not reports['student']:
            st.info('No submitted quiz attempts are available yet.')
            st.stop()

        students = reports['student']
        questions = reports['question']
        flags = discuss_in_class(questions)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Students submitted', len(students))
        c2.metric('Questions analysed', len(questions))
        c3.metric('Mean score', f"{sum(r['Score'] for r in students)/len(students):.1f} / 20")
        c4.metric('Class mean %', f"{sum(r['Percentage'] for r in students)/len(students):.1f}%")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'Student Performance', 'Question Performance', 'Topic / Difficulty',
            'Discuss in Class', 'Wrong / Unanswered'
        ])

        with tab1:
            st.dataframe(students, use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(questions, use_container_width=True, hide_index=True)

        with tab3:
            st.markdown('**Topic performance**')
            st.dataframe(reports['topic'], use_container_width=True, hide_index=True)
            st.markdown('**Difficulty performance**')
            st.dataframe(reports['difficulty'], use_container_width=True, hide_index=True)

        with tab4:
            if flags:
                st.dataframe(flags, use_container_width=True, hide_index=True)
            else:
                st.success('No questions currently meet the discussion thresholds.')
            st.caption('Thresholds are deliberately simple and faculty-reviewable; these are prompts for class discussion, not automated judgements about students.')

        with tab5:
            st.dataframe(reports['errors'], use_container_width=True, hide_index=True)
            st.caption('This table contains every incorrect or unanswered question by student.')

        export_dir = BASE / 'quiz' / 'reports'
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in {
            'student_report.csv': reports['student'],
            'question_report.csv': reports['question'],
            'topic_report.csv': reports['topic'],
            'difficulty_report.csv': reports['difficulty'],
            'student_question_errors.csv': reports['errors'],
            'discuss_in_class.csv': flags,
        }.items():
            write_csv(rows, export_dir / name)

        st.success(f'Reports exported to {export_dir.relative_to(BASE)}')
        st.markdown('**Download faculty reports**')
        for label, filename, rows in [
            ('Student performance', 'student_report.csv', reports['student']),
            ('Question performance', 'question_report.csv', reports['question']),
            ('Topic performance', 'topic_report.csv', reports['topic']),
            ('Difficulty performance', 'difficulty_report.csv', reports['difficulty']),
            ('Student × question errors', 'student_question_errors.csv', reports['errors']),
            ('Discuss in class', 'discuss_in_class.csv', flags),
        ]:
            import io
            buffer = io.StringIO()
            if rows:
                writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
            else:
                buffer.write('No records\n')
            st.download_button(label, buffer.getvalue().encode('utf-8-sig'), filename, 'text/csv', key=f'quiz_report_{filename}')
        st.stop()


    st.subheader('Faculty Evaluation')

    submissions = []

    for folder in SUB_DIR.iterdir():

        if not folder.is_dir():
            continue

        evaluation_file = folder / 'evaluation_results.csv'

        if evaluation_file.exists():
            submissions.append(folder)

    if not submissions:

        st.info('No evaluated submissions available yet.')
        st.stop()

        submissions = sorted(
        submissions,
        key=lambda p: p.name,
        reverse=True
    )

    # -------------------------------------------------
    # Faculty submission overview
    # -------------------------------------------------

    st.divider()
    st.subheader('Submission Overview')

    overview_rows = []

    for folder in submissions:

        submission_metadata = {}

        submission_file = folder / 'submission.csv'

        if submission_file.exists():

            with open(
                submission_file,
                newline='',
                encoding='utf-8'
            ) as f:

                rows = list(
                    csv.DictReader(f)
                )

                if rows:
                    submission_metadata = rows[0]

        evaluation_file = folder / 'evaluation_results.csv'

        with open(
            evaluation_file,
            newline='',
            encoding='utf-8'
        ) as f:

            rows = list(
                csv.DictReader(f)
            )

        verified_count = sum(
            r['verified'] == 'True'
            for r in rows
        )

        review_count = sum(
            r['verified'] == 'False'
            for r in rows
        )

        faculty_count = sum(
            r['verified'] not in ('True', 'False')
            for r in rows
        )

        overview_rows.append({
            'Submission': folder.name,
            'Dataset': submission_metadata.get(
                'assigned_dataset',
                'Not available'
            ),
            'Submitted': submission_metadata.get(
                'submitted_at',
                'Not available'
            ),
            'Verified': verified_count,
            'Review': review_count,
            'Faculty Review': faculty_count
        })

    st.table(overview_rows)

    # -------------------------------------------------
    # Select submission for detailed review
    # -------------------------------------------------

    selected = st.selectbox(
        'Select submission for detailed review',
        submissions,
        format_func=lambda p:
            f"Submission {p.name}"
    )

    # -------------------------------------------------
    # Submission metadata
    # -------------------------------------------------

    submission_file = selected / 'submission.csv'

    metadata = {}

    if submission_file.exists():

        with open(
            submission_file,
            newline='',
            encoding='utf-8'
        ) as f:

            rows = list(csv.DictReader(f))

            if rows:
                metadata = rows[0]

    st.divider()

    st.subheader('Submission Details')

    # -------------------------------------------------
    # Recover SAP ID from submission hash
    # -------------------------------------------------

    sap_id = 'Not available'
    submission_hash = selected.name.split('_')[0]

    for row in DATASET_MAP:
        candidate_hash = hashlib.sha256(
            row.strip().upper().encode()
        ).hexdigest()[:16]

        if candidate_hash == submission_hash:
            sap_id = row
            break

    col1, col2, col3, col4 = st.columns(4)

    col1.write('**SAP ID**')
    col1.write(sap_id)

    col2.write('**Submission ID**')
    col2.write(selected.name)

    col3.write('**Dataset**')
    col3.write(
        metadata.get(
            'assigned_dataset',
            'Not available'
        )
    )

    col4.write('**Submitted**')
    col4.write(
        metadata.get(
            'submitted_at',
            'Not available'
        )
    )

    # -------------------------------------------------
    # Download original submitted ZIP
    # -------------------------------------------------

    submission_zip = selected / 'submission.zip'

    if submission_zip.exists():
        st.download_button(
            'Download Submitted ZIP',
            data=submission_zip.read_bytes(),
            file_name=f'{sap_id}_submission.zip',
            mime='application/zip'
        )
    else:
        st.warning('Original submitted ZIP is not available.')

    # -------------------------------------------------
    # Load evaluation evidence
    # -------------------------------------------------

    with open(
        selected / 'evaluation_results.csv',
        newline='',
        encoding='utf-8'
    ) as f:

        evaluation_rows = list(
            csv.DictReader(f)
        )

    total = len(evaluation_rows)

    verified = sum(
        r['verified'] == 'True'
        for r in evaluation_rows
    )

    review = sum(
        r['verified'] == 'False'
        for r in evaluation_rows
    )

    faculty_review = sum(
        r['verified'] not in ('True', 'False')
        for r in evaluation_rows
    )

    # -------------------------------------------------
    # Evaluation summary
    # -------------------------------------------------

    st.divider()
    st.subheader('Automated Evidence Summary')

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        'Total Checks',
        total
    )

    col2.metric(
        'Verified',
        verified
    )

    col3.metric(
        'Review',
        review
    )

    col4.metric(
        'Faculty Review',
        faculty_review
    )

    # -------------------------------------------------
    # Evidence table
    # -------------------------------------------------

    st.divider()
    st.subheader('Detailed Evidence')

    display_rows = []

    for row in evaluation_rows:

        if row['verified'] == 'True':
            status = '✓ VERIFIED'

        elif row['verified'] == 'False':
            status = '✗ REVIEW'

        else:
            status = '⚠ FACULTY REVIEW'

        display_rows.append({
            'Check': row['check'],
            'Reported': row['reported'],
            'Actual / Reference': row['actual'],
            'Status': status
        })

    st.table(display_rows)
        # -------------------------------------------------
    # -------------------------------------------------
    # Automated provisional scoring
    # -------------------------------------------------

    scoring = None
    scoring_error = None

    submission_zip = selected / 'submission.zip'
    if submission_zip.exists():
        try:
            scoring = score_submission(
                submission_zip,
                evaluation_rows
            )
        except Exception as exc:
            scoring_error = str(exc)
    else:
        scoring_error = 'Original submitted ZIP is not available for content scoring.'

    st.divider()
    st.subheader('Provisional Assessment')

    if scoring is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            'Provisional Score',
            f"{scoring['provisional_score']:.1f} / 30"
        )
        c2.metric(
            'Confidence',
            scoring['confidence']
        )
        c3.metric(
            'Review Priority',
            scoring['review_priority']
        )

        st.caption(
            'The provisional score is conservative and evidence-based. '
            'Faculty review remains the final authority.'
        )

        rubric_rows = []
        for component, maximum in scoring['rubric_scores'].items():
            rubric_rows.append({
                'Rubric component': component,
                'Provisional marks': scoring['rubric_scores'][component],
                'Maximum': maximum,
            })
        st.dataframe(
            rubric_rows,
            use_container_width=True,
            hide_index=True
        )

        if scoring.get('issues'):
            st.warning('Potential scoring deductions identified by automated evidence')
            issue_rows = []
            for issue in scoring['issues']:
                issue_rows.append({
                    'Rubric': issue['rubric'],
                    'Category': issue['category'],
                    'Deduction': issue['deduction'],
                    'Observation': issue['observation'],
                })
            st.dataframe(
                issue_rows,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success('No substantive automated deduction was identified.')

        if scoring.get('review_flags'):
            st.info('Faculty review flags — these do not automatically reduce the provisional score')
            flag_rows = []
            for flag in scoring['review_flags']:
                flag_rows.append({
                    'Rubric': flag['rubric'],
                    'Observation': flag['observation'],
                    'Why it matters': flag['why_it_matters'],
                    'Suggested action': flag['prescription'],
                })
            st.dataframe(
                flag_rows,
                use_container_width=True,
                hide_index=True
            )

        with st.expander('Automated faculty comments', expanded=bool(scoring.get('issues') or scoring.get('review_flags'))):
            automated_comments = scoring.get('faculty_comments', '').strip()
            if automated_comments:
                st.write(automated_comments)
            else:
                st.write('No automated faculty comments were generated.')
    else:
        st.error(
            'Provisional scoring could not be generated for this submission.'
        )
        st.caption(scoring_error or 'Unknown scoring error.')

    # -------------------------------------------------
    # Faculty Review
    # -------------------------------------------------

    review_file = selected / 'faculty_review.csv'
    existing_status = 'Pending'
    existing_notes = ''
    existing_score = None
    existing_feedback = ''

    if review_file.exists():
        with open(
            review_file,
            newline='',
            encoding='utf-8'
        ) as f:
            review_rows = list(
                csv.DictReader(f)
            )

            if review_rows:
                existing_status = review_rows[0].get(
                    'review_status',
                    'Pending'
                )
                existing_notes = review_rows[0].get(
                    'faculty_notes',
                    ''
                )
                raw_score = review_rows[0].get(
                    'final_score',
                    ''
                )
                try:
                    existing_score = float(raw_score)
                except (TypeError, ValueError):
                    existing_score = None
                existing_feedback = review_rows[0].get(
                    'student_feedback',
                    ''
                )

    review_status = st.selectbox(
        'Review Status',
        ['Pending', 'Reviewed'],
        index=(
            1
            if existing_status == 'Reviewed'
            else 0
        )
    )

    default_notes = existing_notes
    if not default_notes and scoring is not None:
        default_notes = scoring.get('faculty_comments', '')

    faculty_notes = st.text_area(
        'Faculty Notes',
        value=default_notes,
        height=180,
        placeholder='Edit or add your faculty observations and final judgement...'
    )

    default_score = existing_score
    if default_score is None and scoring is not None:
        default_score = float(scoring['provisional_score'])
    if default_score is None:
        default_score = 0.0

    final_score = st.number_input(
        'Final Score (/30)',
        min_value=0.0,
        max_value=30.0,
        value=default_score,
        step=0.5
    )

    default_feedback = existing_feedback
    if not default_feedback and scoring is not None:
        default_feedback = scoring.get('student_feedback', '')

    st.subheader('Suggested Student Feedback')
    st.caption(
        'This is a draft. Faculty should review/edit it before sharing with students.'
    )
    student_feedback = st.text_area(
        'Student Feedback',
        value=default_feedback,
        height=280,
        label_visibility='collapsed'
    )

    if st.button(
        'Save Faculty Review',
        type='primary'
    ):
        with open(
            review_file,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                'review_status',
                'faculty_notes',
                'final_score',
                'student_feedback',
                'reviewed_at'
            ])
            writer.writerow([
                review_status,
                faculty_notes,
                final_score,
                student_feedback,
                datetime.now().isoformat()
            ])

        st.success(
            'Faculty review, final score and draft student feedback saved successfully.'
        )


    # -------------------------------------------------
    # Consolidated final marks export
    # -------------------------------------------------

    st.divider()
    st.subheader('Faculty Marks Export')
    st.caption(
        'Downloads one CSV containing SAP ID, final score and the saved faculty review for every evaluated submission.'
    )

    marks_rows = []
    for folder in submissions:
        metadata = {}
        submission_file = folder / 'submission.csv'
        if submission_file.exists():
            with open(
                submission_file,
                newline='',
                encoding='utf-8'
            ) as f:
                rows = list(csv.DictReader(f))
                if rows:
                    metadata = rows[0]

        submission_hash = folder.name.split('_')[0]
        sap_id_export = 'Not available'
        for candidate in DATASET_MAP:
            candidate_hash = hashlib.sha256(
                candidate.strip().upper().encode()
            ).hexdigest()[:16]
            if candidate_hash == submission_hash:
                sap_id_export = candidate
                break

        review_data = {}
        review_file_export = folder / 'faculty_review.csv'
        if review_file_export.exists():
            with open(
                review_file_export,
                newline='',
                encoding='utf-8'
            ) as f:
                review_rows_export = list(csv.DictReader(f))
                if review_rows_export:
                    review_data = review_rows_export[0]

        marks_rows.append({
            'SAP ID': sap_id_export,
            'Dataset': metadata.get('assigned_dataset', 'Not available'),
            'Submission ID': folder.name,
            'Review Status': review_data.get('review_status', 'Pending'),
            'Final Score': review_data.get('final_score', ''),
            'Faculty Notes': review_data.get('faculty_notes', ''),
            'Student Feedback': review_data.get('student_feedback', ''),
            'Reviewed At': review_data.get('reviewed_at', ''),
        })

    marks_buffer = io.StringIO()
    if marks_rows:
        marks_writer = csv.DictWriter(
            marks_buffer,
            fieldnames=list(marks_rows[0].keys())
        )
        marks_writer.writeheader()
        marks_writer.writerows(marks_rows)
    else:
        marks_buffer.write('No evaluated submissions\n')

    st.download_button(
        'Download Final Marks CSV',
        data=marks_buffer.getvalue().encode('utf-8-sig'),
        file_name='final_marks.csv',
        mime='text/csv',
        key='final_marks_csv_export'
    )

    # Governance reminder
    # -------------------------------------------------

    st.divider()

    st.warning(
        'Automated evidence only. '
        'This is not a final grade or faculty feedback. '
        'Faculty judgement remains required.'
    )

    st.stop()
student_id = st.text_input(
    'Student ID',
    placeholder='Enter your official student ID'
).strip().upper()

if student_id:
    activity = st.radio(
        'Choose activity',
        ['ML Assignment', 'ML Quiz'],
        horizontal=True
    )

    if activity == 'ML Quiz':
        _quiz_student_view(student_id)
        st.stop()

if not student_id:
    st.stop()

if DATASET_MAP:

    assigned_dataset = DATASET_MAP.get(student_id)

    if assigned_dataset is None:

        st.error(
            'Student ID not found in the assignment allocation.'
        )

        st.stop()

else:

    st.error(
        'Assignment allocation is not configured. '
        'Please contact the faculty administrator.'
    )

    st.stop()

base_dataset = assigned_dataset.split('_V')[0] + '.csv' if '_V' in assigned_dataset else assigned_dataset
meta = DATASET_META.get(base_dataset, {'title':'Individual ML Case','type':'ML'})
st.success(f"Your assigned case: {meta['title']}")
st.write(f"**ML problem type:** {meta['type']}")

dataset_id = assigned_dataset.replace(
    'ML_Dataset_', ''
).replace(
    '.csv', ''
)

st.info(f"**Your Dataset ID: {dataset_id}**")

path = DATA_DIR / ('variants' if '_V' in assigned_dataset else '') / assigned_dataset
if path.exists():
    st.download_button('Download your dataset', data=path.read_bytes(), file_name='your_assigned_dataset.csv', mime='text/csv')
else:
    st.warning("Dataset file is not installed. Place the faculty CSVs inside the 'datasets' folder.")

st.divider()
st.subheader('Submit your assignment')

st.markdown(
    'Upload your completed **submission ZIP** containing the '
    '`Results_Template.xlsx`, analysis file, and `Executive_Summary.pdf`.'
)

zip_file = st.file_uploader(
    'Upload submission ZIP',
    type=['zip']
)

confirm = st.checkbox(
    'I confirm that these files represent my submitted work.'
)

if st.button('Submit Assignment', type='primary'):

    if not zip_file:
        st.error('Please upload your submission ZIP.')
        st.stop()

    if not confirm:
        st.error('Please confirm your submission.')
        st.stop()

    student_hash = hashlib.sha256(
        student_id.encode()
    ).hexdigest()[:16]

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )

    folder = SUB_DIR / f'{student_hash}_{timestamp}'
    folder.mkdir(
        parents=True,
        exist_ok=False
    )

    zip_path = folder / 'submission.zip'

    zip_path.write_bytes(
        zip_file.getbuffer()
    )

    st.write('Checking submission structure...')

    if not validate_submission(zip_path):

        st.error(
            'Submission is incomplete. '
            'Please correct the issues above and resubmit.'
        )

        shutil.rmtree(
            folder,
            ignore_errors=True
        )

        st.stop()
        st.write('Running automated evaluation...')

    dataset_id = assigned_dataset.replace(
        'ML_Dataset_', ''
    ).replace(
        '.csv', ''
    )

    evaluation_results = evaluate_submission(
        zip_path,
        dataset_id,
        return_results=True
    )

    results_file = folder / 'evaluation_results.csv'

    with open(
        results_file,
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        w = csv.writer(f)

        w.writerow([
            'check',
            'reported',
            'actual',
            'verified'
        ])

        for result in evaluation_results:

            w.writerow([
                result.get('check'),
                result.get('reported'),
                result.get('actual'),
                result.get('verified')
            ])
    with open(
        folder / 'submission.csv',
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        w = csv.writer(f)

        w.writerow([
            'student_id_hash',
            'assigned_dataset',
            'submitted_at'
        ])

        w.writerow([
            student_hash,
            assigned_dataset,
            datetime.now().isoformat()
        ])

    st.success(
        'Submission received successfully.'
    )

    st.info(
        'Your submission has passed the structural validation '
        'and is ready for evaluation.'
    )
st.divider(); st.caption('Prototype v0.1 — faculty development version')
