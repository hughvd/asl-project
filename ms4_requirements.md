[MS4] Final Modeling + Deliverables [4/24-5/12]

Hard deadline for the three graded MS4 deliverables (report, video, code): Tuesday, May 12, 2026 at 9:59pm Eastern Time (America/New_York; in May this is EDT, UTC−04:00). The Canvas submission link auto-closes at 10:00pm ET sharp. There are no late days. Your upload must be fully completed and timestamped by 9:59pm ET — a submission that starts before 9:59pm but finishes after 10:00pm will not count. Start uploading early. If something goes wrong, submit what you have rather than nothing at all. The peer-evaluation assignment is due 24 hours later.

OPTIONAL: Feedback Meeting with TF (before final submission) — by Friday 5/8. The teaching staff strongly believes the quality of your final deliverables will benefit from a check-in on your progress since MS3. Coordinate with your TF to schedule, and come prepared with specific questions.

📦 Milestone 4: Final Modeling + Deliverables [68 pts]

MS4 is the final stage of your group project. It is worth 68 points (≈68% of your total project grade), distributed across three graded deliverables plus a required peer-evaluation form:

Final Report — 35 pts
Video Presentation — 25 pts
Code Notebook — 8 pts
Peer Evaluations — required to pass MS4 (see below)
Citation policy: cite all use of generative AI agents (if used) as well as any outside resources used in your .ipynb, report PDF, or video. See here and here for citing expectations, as well as the course syllabus for our policies. Violations will result in a reduction of your group's score with no option of a regrade. 

----------------------------------------------------------
🏆 Final Report — 35 pts

A written, paper-style report communicating your final modeling work and findings to a reader who has not seen your prior milestones.

Length & format

Target length: 2000–2500 words (roughly 5 pages of body text). This is a soft target, not a hard cap — modest overruns (a few hundred words) will not cost points if the writing is tight, but a 4000-word report will be marked down on the writing rubric. Aim for 2000–2500. 
What counts: body prose of all required sections, including figure and table captions and any footnotes. What does not count: the title page (title, names, Canvas number), section headings, references/bibliography, and the appendix.
Submission format: a single PDF, uploaded to Canvas by one group member on behalf of the team. Filename suggestion: cs1090b_ms4_report_group<NN>.pdf.
Template (recommended): the NeurIPS 2026 LaTeX template (single-column, easier to read than 2-column formats). Any clean academic format is acceptable; please avoid 2-column layouts, screenshots of code, and illegibly small figure text.
An appendix is permitted and encouraged for supplementary visuals, tables, or methodological detail that did not fit in the body. The appendix does not count against your word budget but is also not required reading for the grader.
Required structure

Organize the body of the report along the lines of a standard scientific paper. The exact section names are flexible, but each of the following must be clearly addressed:

Title, group Canvas number, group member names (on the first page).
Background & Motivation — why this problem matters, who is affected, what decision or impact your work supports.
Problem Statement — your final, refined research question, target/output, and unit of observation.
Data & EDA — concise description of the data and the EDA findings that materially influenced modeling decisions. (Do not re-paste your full MS2/MS3 EDA — distill it.)
Methods & Models — your final modeling pipeline, models considered, justification for choices, training details (data splits, tuning, hyperparameters).
Results & Interpretation — quantitative results, comparison to your MS3 baseline, interpretation of what the model learned (feature importance, error analysis, etc.).
Conclusions & Discussion — what you can and cannot claim, limitations, future work.
Broader Impact — briefly describe and discuss the broader impact of your project in the bigger picture (e.g., who benefits, what risks or potential harms come with deployment, how this work connects to wider questions in the domain).
References — every outside source (datasets, papers, code repositories, generative-AI assistance).
AC209B groups: Per the syllabus, groups containing one or more 209B students must use a method or approach not explicitly covered in class and must communicate an understanding of that method and its applicability to your problem. The Methods and Discussion sections of the report are the primary place where this is evaluated, with the presentation as supporting evidence. If you are unsure whether a particular method qualifies, check with your TF or the instructor before MS4 submission rather than after.

What we'll grade on

Clarity & motivation — the reader understands what you did and why it matters.
Data & EDA depth — preprocessing, exploration, and key findings are concise but substantive.
Modeling justification — model choices and pipeline decisions are tied to your data and question.
Results & interpretation — quantitative results are presented honestly, compared to baseline, and interpreted (not just reported).
Writing & visuals — clean prose, readable figures with axis labels and captions, minimal typos.
Citations — all sources, code, and AI assistance are cited.
----------------------------------------------------------
📽️ Video Presentation — 25 pts

A recorded presentation that walks a viewer through your project from motivation to results in 6 minutes or less. This recording replaces a typical in-person talk; treat it like a conference talk, not an unedited screencast.

Format & submission

Hard time limit: 6 minutes. Anything beyond 6:00 will not be reviewed; the grader will stop the video at the 6-minute mark, and content after that does not count. The time-discipline component reflects whether you fit your material into the limit.
Recording must play at 1.0× normal speed. Do not speed up the recording (or your speech via a post-processing tool) to fit more content into 6 minutes — recordings that are obviously sped up will be treated as exceeding the time limit.
All team members must appear on camera and speak. Distribute the talk so every member presents a meaningful portion (not one slide of "hi, I'm X"). If a team member needs an accommodation (camera-off, etc.), email the instructor and your TF before recording — do not surprise us at submission time.
Submission: upload the recording to YouTube as an unlisted video (preferred) or to Google Drive, and submit the shareable URL as a comment on the Canvas submission. The link must be viewable without sign-in by anyone with the URL, and must remain accessible until June 1, 2026 (after final grades are posted). Videos are viewed by your grader and 2–3 other staff members for grading purposes only and will not be distributed outside the course.
Required arc

Your video should cover, roughly in order:

Introductions of all team members.
Background, motivation, and problem statement.
Data & key EDA findings (visual, not exhaustive).
Modeling approach and training details (time taken, epochs, batch size, learning rate, etc., where relevant).
Results & interpretation.
Conclusions and future work.
Note: it is acceptable to gloss over deep details and say "see the report or notebook for details on X." But viewers should leave with a clear understanding of your problem, your methods and why you chose them, your results, and what future work would look like.

Pavlos' 10 commandments for presentations

Practice three times. Don't read from a script — practice your presentation at least three times to develop a natural flow and pay attention to your intonation.
Tell a clear story. Your slides should be a coherent narrative, not a collection of disconnected results. Every slide must move the story forward — no "NPC" slides that just sit there, and be sure to talk to each slide.
One idea per slide; fewer words. Each slide should make one main point. Slides are not paragraphs — split text-heavy content across multiple slides or balance text with white space.
Use the rule of three. Where you can, organize ideas in groups of three (three takeaways, three reasons, three steps). It keeps slides balanced and easier to follow.
Make plots readable. Use large enough fonts on axes and tick labels — assume the viewer cannot zoom. Label everything clearly: axes (with units), titles, and legends.
Consistent typography. Keep font sizes consistent throughout, use no more than three different fonts, highlight only essential keywords, and avoid unnecessary capitalization.
Audio & visual polish. Check the audio quality and remove background noise. Edit and smooth the transitions between slides, and consider the aesthetics (color use, etc.).
Include slide numbers. They help your teaching team — and you — reference specific content during questions and grading.
Show up as a team. Introduce your teammates, use "we" instead of "I" to show the presentation is a team effort, and make sure each presenter has an appropriate on-camera background.
Land it on time, with energy. Stay within the time limit and convey genuine excitement about your project — engage with the audience and relay the highlights of your effort.
What we'll grade on

Technical content — accurate, substantive description of your problem, methods, and results.
Storytelling & organization — sections connect, the argument flows, slides support the narrative.
Verbal clarity — pace, intonation (e.g., not monotone reading from a script), audio quality, and equitable participation across team members.
Slide quality — readable text and figures, minimal clutter, consistent style.
Time discipline — content delivered within the 6-minute limit at a normal speaking pace.
----------------------------------------------------------
📒 Code Notebook — 8 pts

Notebook scope: not all of your code needs to live in a single notebook, but your project's main pipeline and final results must be demonstrated end-to-end in one designated "main" notebook. The main notebook must contain executable cells that load the data, run the modeling pipeline, and produce the final reported results — it cannot be a thin wrapper that just imports and prints. Helper functions, dataset utilities, training scripts, and side experiments are welcome to live in additional .py modules or supporting .ipynb files in your submission; the main notebook is allowed (and encouraged) to import from these. The grader will read and run the main notebook.

Requirements

Name the main notebook clearly, e.g. cs1090b_ms4_main_group<NN>.ipynb. Any other notebooks or scripts should have descriptive names.
The main notebook should be readable enough to be handed to a colleague who has not seen your project — use clear section headings, prose explanations between code cells, and inline comments / docstrings.
Use markdown style headings (e.g., ## Heading) for all major sections of your notebook. This will generate a nice, interactive table of contents when the notebook is viewed in Colab or Jupyter Lab.
Include the group's Canvas number and member names at the top of the notebook.
Document library dependencies (a requirements.txt, environment.yml, or an explicit %pip install cell at the top — Colab-style — is fine) and any non-obvious environment configuration (GPU, data paths, API keys, etc.). If you use in-line pip be sure to use the --quite flag to suppress the noisy output.
Reproducibility: "Restart kernel + Run All" should succeed on a machine that has the documented dependencies installed and the data available at the documented path (use relative paths, i.e., ./data/file.csv, not /works/on/my/machine/data/file.csv). Internet access is allowed (e.g. to download a pretrained model). If your full training run takes hours, it is OK for the notebook to load a pre-saved checkpoint or cached intermediate file as long as the code that produced it is also present (in the main notebook or an imported script) and the load path is explicit.
Submission — submit either a zip or a text file with a public repo commit link, not both:
Default option (zip): bundle the main notebook + any supporting .py / .ipynb files into a single .zip, ≤ 50 MB, and upload to Canvas. Do not include raw datasets larger than a few hundred MB; describe how to obtain them instead.
Alternative option (commit link): if your zip would exceed 50 MB, instead submit a link to a specific commit on a public Git repository (e.g. GitHub) containing the same files. The repo must remain public and unmodified until June 1, 2026 (after final grades are posted) — do not force-push or rename branches. This must be a link to a specific commit so we can verify that the files have not changed after the deadline.
What we'll grade on

Organization & clarity — TOC, section headings, prose explanations.
Code quality — readable, commented, free of dead code and stray debugging cells.
Reproducibility — runs cleanly from a fresh kernel given the documented environment.
----------------------------------------------------------
🤝 Peer Evaluations — required to pass MS4

A self & peer evaluation will be released as a separate Canvas assignment. The form is released after the MS4 deadline closes and is due:

Peer evaluation due: Wednesday, May 13, 2026 at 9:59pm Eastern Time (24 hours after the MS4 deadline). Each group member must complete the form individually.

"Submitted" means a fully completed form posted to the Canvas peer-eval assignment by the due time — drafts and partially completed forms do not count.

Failure to submit your individual peer evaluation by the deadline will result in an Incomplete on MS4 for that student. The consequence is per-individual: one student's failure to submit does not affect their teammates' MS4 grade.

----------------------------------------------------------
✅ Submission checklist

By Tuesday, May 12, 2026, 9:59pm Eastern Time (Canvas auto-closes at 10:00pm ET), submit on the MS4 Canvas assignment:

One final report PDF (one per group, uploaded by one group member).
One video link (YouTube unlisted preferred), placed in the assignment URL/text field and as a comment on the submission (one per group, viewable without sign-in, accessible until June 1, 2026).
One code submission — main notebook plus any supporting files, as a single .zip or a link to a public Git repository (pick one; one per group).
And then by Wednesday, May 13, 2026 at 9:59pm Eastern Time (24 hours later):

One peer evaluation per student, on the separate Canvas peer-evaluation assignment (individual; required to pass MS4 — see above).
----------------------------------------------------------
📋 General expectations

Across all three graded deliverables, we are looking for the same underlying qualities we evaluated in MS2 and MS3: quality and thoroughness of thinking over visual polish. A clean, modest deliverable that demonstrates rigorous reasoning will outscore a flashy one that doesn't.

🥳 And: be sure to celebrate! 🎉

You will have completed a substantial, end-to-end data-science project as a team.