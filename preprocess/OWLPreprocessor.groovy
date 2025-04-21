@Grab(group='org.slf4j', module='slf4j-api', version='1.7.36')
@Grab(group='org.slf4j', module='slf4j-simple', version='1.7.36') // Added for simple logging
@Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3') // Kept if needed elsewhere, not used in this script
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.5.26') // Updated version
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.5.26') // Updated version
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.5.26') // Updated version
@Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.5.26') // Updated version
@Grab(group='net.sourceforge.owlapi', module='owlapi-oboformat', version='4.5.26') // Added for potential OBO parsing needs
@Grab(group='net.sourceforge.owlapi', module='owlapi-tools', version='4.5.26') // Added for profiles
@Grab(group='net.sourceforge.owlapi', module='owlapi-rio', version='4.5.26') // Added for RDF/XML parsing etc.


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

// --- Basic Setup ---
// Configure logging to avoid excessive output from OWLAPI/dependencies
System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "warn")

// --- Argument Handling ---
if (args.length == 0) {
    println "Usage: groovy OWLPreprocessor.groovy <path/to/ontology.owl> [output/path/nnf_ontology.owl]"
    println "       Outputs statistics and NNF axioms (Functional Syntax) to console."
    println "       If an output path is provided, saves the NNF ontology there as well."
    return
}
String ontologyPath = args[0]
File ontologyFile = new File(ontologyPath)
String outputPath = (args.length > 1) ? args[1] : null

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


// --- Output NNF Axioms in Functional Syntax (Console) ---
println "--- NNF Axioms (Functional Syntax) ---"
// Setup renderer
OWLFunctionalSyntaxRenderer renderer = new OWLFunctionalSyntaxRenderer()
StringDocumentTarget target = new StringDocumentTarget() // To capture output as string

// Use a PrefixManager for cleaner output if prefixes are defined
DefaultPrefixManager pm = new DefaultPrefixManager(null, null, ont.getFormat().asPrefixOWLDocumentFormat().getPrefixName2PrefixMap())
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

// --- Optional: Save NNF Ontology to File ---
if (outputPath) {
    try {
        println "\nSaving NNF ontology to: ${outputPath}"
        // Create a new ontology containing only the NNF axioms
        OWLOntology nnfOntology = manager.createOntology(nnfAxioms)

        // Set prefixes for the output format
        FunctionalSyntaxDocumentFormat functionalSyntaxFormat = new FunctionalSyntaxDocumentFormat()
        functionalSyntaxFormat.copyPrefixesFrom(pm)
        functionalSyntaxFormat.setPrefix(":", pm.getDefaultPrefix()) // Ensure default prefix is set

        // Save the NNF ontology
        File outputFile = new File(outputPath)
        manager.saveOntology(nnfOntology, functionalSyntaxFormat, IRI.create(outputFile.toURI()))
        println "NNF ontology saved successfully."
    } catch (Exception e) {
        println "Error saving NNF ontology to file: ${e.message}"
        // e.printStackTrace()
    }
}
