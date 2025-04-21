@Grab(group='org.slf4j', module='slf4j-api', version='1.7.36')
@Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3')
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.5.20')
@Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.5.20')


import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.formats.FunctionalSyntaxDocumentFormat
import org.semanticweb.owlapi.model.*
import org.semanticweb.owlapi.profiles.*
import org.semanticweb.owlapi.util.*
import org.semanticweb.owlapi.io.*
import org.semanticweb.owlapi.model.parameters.Imports
import org.semanticweb.owlapi.search.*
import org.semanticweb.owlapi.normalform.*
import org.semanticweb.owlapi.util.mansyntax.ManchesterOWLSyntaxParser // Needed? Maybe not directly
import org.semanticweb.owlapi.manchestersyntax.renderer.ManchesterOWLSyntaxObjectRenderer // For potential rendering options
import org.semanticweb.owlapi.functional.renderer.OWLFunctionalSyntaxRenderer // For functional syntax output
import java.nio.file.Paths
import java.nio.file.Path

// --- Basic Setup ---
// Configure logging to avoid excessive output from OWLAPI/dependencies
System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "warn")

// --- Argument Handling ---
def usage() {
    println """
Usage: groovy OWLPreprocessor.groovy <path/to/ontology.owl> [options]

Options:
  <output/path/nnf_ontology.owl>   (Optional) Path to save the combined NNF ontology.
                                     If provided without --split, saves all NNF axioms here.
  --split <output/base/path>       (Optional) Split NNF axioms into ABox, TBox, RBox files.
                                     Generates <base_path>_abox.owl, <base_path>_tbox.owl, etc.
                                     If used, the single output path argument is ignored.

Description:
  Loads an OWL ontology, prints statistics, converts logical axioms to
  Negation Normal Form (NNF), and outputs the NNF axioms.
  By default, NNF axioms are printed to the console (Functional Syntax).
  Use output options to save NNF axioms to files.
"""
}

if (args.length == 0 || args[0] in ['-h', '--help']) {
    usage()
    return
}

String ontologyPath = args[0]
File ontologyFile = new File(ontologyPath)
String singleOutputPath = null
String splitBasePath = null
boolean splitOutput = false

// Parse options
int i = 1
while (i < args.length) {
    if (args[i] == '--split' && i + 1 < args.length) {
        splitOutput = true
        splitBasePath = args[i+1]
        i += 2
    } else if (!args[i].startsWith('--') && singleOutputPath == null && !splitOutput) {
        // Assume it's the single output path if not splitting and not already set
        singleOutputPath = args[i]
        i += 1
    } else {
        println "Error: Invalid argument or combination: ${args[i]}"
        usage()
        return
    }
}

if (splitOutput && singleOutputPath != null) {
    println "Warning: Single output path (${singleOutputPath}) ignored because --split option is used."
    singleOutputPath = null // Ensure single output is not processed
}


if (!ontologyFile.exists()) {
    println "Error: Ontology file not found at ${ontologyPath}"
    return
}

// --- OWLAPI Setup ---
OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLDataFactory fac = manager.getOWLDataFactory()
OWLOntology ont

// --- Load Ontology ---
try {
    println "Loading ontology from: ${ontologyPath}"
    // Increase robustness by allowing different parsers based on format
    OWLOntologyLoaderConfiguration config = new OWLOntologyLoaderConfiguration()
    config = config.setMissingImportHandlingStrategy(MissingImportHandlingStrategy.SILENT) // Handle missing imports gracefully
    ont = manager.loadOntologyFromOntologyDocument(ontologyFile, config)
    println "Ontology loaded successfully. IRI: ${ont.getOntologyID().getOntologyIRI().orNull()}"
} catch (Exception e) {
    println "Error loading ontology: ${e.message}"
    // e.printStackTrace() // Uncomment for detailed stack trace
    return
}

// --- Ontology Statistics ---
println "\n--- Ontology Statistics ---"
// Profile Check
OWL2DLProfile profile = new OWL2DLProfile()
OWLProfileReport report = profile.checkOntology(ont)
println "OWL 2 DL Profile Compliant: ${report.isInProfile() ? 'Yes' : 'No'}"
if (!report.isInProfile()) {
    println "Violations (${report.getViolations().size()}):"
    // Print only first few violations for brevity
    report.getViolations().take(5).each { violation -> println "- ${violation.toString().replaceAll("\\s+", " ").take(150)}..." }
    if (report.getViolations().size() > 5) println "  ... and more."
}

// Expressivity (List axiom types present)
Set<AxiomType<?>> axiomTypes = ont.getAxiomTypes(Imports.INCLUDED)
println "Axiom Types Present (${axiomTypes.size()}): ${axiomTypes.collect { it.getName() }.sort().join(', ')}"

// Counts
println "Logical Axiom Count: ${ont.getLogicalAxiomCount(Imports.INCLUDED)}"
println "TBox Axiom Count: ${ont.getTBoxAxioms(Imports.INCLUDED).size()}"
println "ABox Axiom Count: ${ont.getABoxAxioms(Imports.INCLUDED).size()}"
println "RBox Axiom Count: ${ont.getRBoxAxioms(Imports.INCLUDED).size()}"
println "Class Count: ${ont.getClassesInSignature(Imports.INCLUDED).size()}"
println "Object Property Count: ${ont.getObjectPropertiesInSignature(Imports.INCLUDED).size()}"
println "Data Property Count: ${ont.getDataPropertiesInSignature(Imports.INCLUDED).size()}"
println "Individual Count: ${ont.getIndividualsInSignature(Imports.INCLUDED).size()}"
println "-------------------------\n"


// --- NNF Conversion ---
println "--- Converting to Negation Normal Form (NNF) ---"
NegationalNormalFormConverter nnfConverter = new NegationalNormalFormConverter(fac)
List<OWLAxiom> nnfAxioms = []
int skippedAxioms = 0
int originalAxiomCount = ont.getLogicalAxiomCount(Imports.INCLUDED)

ont.getLogicalAxioms(Imports.INCLUDED).each { axiom ->
    try {
        // Convert axiom to NNF.
        OWLAxiom nnfAxiom = nnfConverter.convertToNNF(axiom)
        nnfAxioms.add(nnfAxiom)
    } catch (Exception e) {
        // Catch potential errors during conversion for specific axioms
        println "Warning: Could not convert axiom to NNF: ${axiom.toString().take(100)}... Error: ${e.message}"
        skippedAxioms++
    }
}
println "NNF conversion attempted on ${originalAxiomCount} logical axioms."
println "${nnfAxioms.size()} axioms resulted from NNF conversion."
if (skippedAxioms > 0) {
    println "${skippedAxioms} axioms could not be converted due to errors."
}
println "------------------------------------------------\n"

// --- Prepare Prefix Manager for Output ---
DefaultPrefixManager pm = new DefaultPrefixManager(null, null, ont.getFormat().asPrefixOWLDocumentFormat().getPrefixName2PrefixMap())
// Ensure default prefix is set if one exists
String defaultPrefix = pm.getDefaultPrefix()
if (defaultPrefix == null) {
    // Attempt to find a common base IRI to use as default prefix if none is set
    Optional<IRI> ontIRI = ont.getOntologyID().getOntologyIRI()
    if (ontIRI.isPresent()) {
        String iriStr = ontIRI.get().toString()
        // Simple heuristic: use the IRI up to the last # or /
        int lastHash = iriStr.lastIndexOf('#')
        int lastSlash = iriStr.lastIndexOf('/')
        int splitPoint = Math.max(lastHash, lastSlash)
        if (splitPoint > 0) {
            defaultPrefix = iriStr.substring(0, splitPoint + 1)
            pm.setDefaultPrefix(defaultPrefix)
            println "INFO: Setting default prefix for output: ${defaultPrefix}"
        }
    }
}
if (defaultPrefix != null && !defaultPrefix.endsWith(":") && !defaultPrefix.endsWith("/")) {
     // Ensure the default prefix ends appropriately if we set it manually or copy it
     // This might be overly cautious depending on OWLAPI version behavior
     // pm.setDefaultPrefix(defaultPrefix + "#"); // Or use '/' - consistency is key
}


// --- Helper Function to Save Axioms ---
def saveAxiomsToFile(List<OWLAxiom> axiomsToSave, String filePath, OWLOntologyManager mgr, PrefixManager prefixMgr) {
    if (axiomsToSave.isEmpty()) {
        println "INFO: No axioms to save for ${Paths.get(filePath).getFileName()}."
        return
    }
    OWLOntology tempOntology = null
    try {
        println "Saving ${axiomsToSave.size()} axioms to: ${filePath}"
        tempOntology = mgr.createOntology(new HashSet<>(axiomsToSave)) // Use Set for createOntology

        // Set up format and prefixes
        FunctionalSyntaxDocumentFormat functionalSyntaxFormat = new FunctionalSyntaxDocumentFormat()
        functionalSyntaxFormat.copyPrefixesFrom(prefixMgr)
        if (prefixMgr.getDefaultPrefix() != null) {
             functionalSyntaxFormat.setPrefix(":", prefixMgr.getDefaultPrefix()) // Ensure default prefix is explicitly set for format
        }


        // Save the ontology
        File outputFile = new File(filePath)
        outputFile.getParentFile().mkdirs() // Ensure directory exists
        mgr.saveOntology(tempOntology, functionalSyntaxFormat, IRI.create(outputFile.toURI()))
        println "Saved successfully: ${filePath}"
    } catch (Exception e) {
        println "Error saving ontology to file ${filePath}: ${e.message}"
        // e.printStackTrace()
    } finally {
        // Clean up the temporary ontology
        if (tempOntology != null) {
            mgr.removeOntology(tempOntology)
        }
    }
}


// --- Output / Save NNF Axioms ---

if (splitOutput) {
    println "--- Splitting NNF Axioms into ABox, TBox, RBox files ---"
    List<OWLAxiom> nnfAboxAxioms = []
    List<OWLAxiom> nnfTboxAxioms = []
    List<OWLAxiom> nnfRboxAxioms = []

    nnfAxioms.each { nnfAxiom ->
        if (nnfAxiom.isAboxAxiom()) {
            nnfAboxAxioms.add(nnfAxiom)
        }
        // Note: TBox includes RBox according to OWLAPI isTboxAxiom() definition.
        // We check RBox first to separate them cleanly.
        if (nnfAxiom.isRboxAxiom()) {
            nnfRboxAxioms.add(nnfAxiom)
        } else if (nnfAxiom.isTboxAxiom()) { // Catches remaining TBox axioms (like SubClassOf)
            nnfTboxAxioms.add(nnfAxiom)
        }
        // Axioms like Declarations might not fall into A/T/RBox, handle if necessary
        // else { println "DEBUG: Axiom not classified as ABox/TBox/RBox: ${nnfAxiom}" }
    }

    println "Categorized NNF axioms: ABox(${nnfAboxAxioms.size()}), TBox(${nnfTboxAxioms.size()}), RBox(${nnfRboxAxioms.size()})"

    // Define output filenames
    String aboxPath = "${splitBasePath}_abox.owl"
    String tboxPath = "${splitBasePath}_tbox.owl"
    String rboxPath = "${splitBasePath}_rbox.owl"

    // Save each category
    saveAxiomsToFile(nnfAboxAxioms, aboxPath, manager, pm)
    saveAxiomsToFile(nnfTboxAxioms, tboxPath, manager, pm)
    saveAxiomsToFile(nnfRboxAxioms, rboxPath, manager, pm)

    println "---------------------------------------------------------"

} else {
    // Original behavior: Print to console and optionally save to single file

    println "--- NNF Axioms (Functional Syntax) ---"
    // Setup renderer
    OWLFunctionalSyntaxRenderer renderer = new OWLFunctionalSyntaxRenderer()
    StringDocumentTarget target = new StringDocumentTarget() // To capture output as string
    renderer.setPrefixManager(pm)

    nnfAxioms.each { nnfAxiom ->
        // Render each axiom
        target.reset() // Clear the target for the next axiom
        try {
            // Render axiom using the ontology's prefix manager if available
            renderer.render(ont, nnfAxiom, target)
            println target.toString().trim() // Print the rendered axiom
        } catch (Exception e) {
            println "Error rendering NNF axiom: ${nnfAxiom.toString().take(100)}... Error: ${e.message}"
        }
    }
    println "--------------------------------------"

    // Optional: Save combined NNF Ontology to File
    if (singleOutputPath) {
        saveAxiomsToFile(nnfAxioms, singleOutputPath, manager, pm)
    }
}

println "Processing finished."
